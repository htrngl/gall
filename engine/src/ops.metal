// everything that is not a weight mat-vec. params are a uint array p
// (floats bit-cast); a = f32 activations, s = small f32 weights,
// kv = f16 cache, cs = conv state, id = expert ids / argmax.
// one-threadgroup kernels run T = 768 threads: thread t owns elements
// t, t+T, t+2T of a hidden vector; simdgroup sg owns head sg.
#include <metal_stdlib>
using namespace metal;
#define N 2304
#define T 768
#define C 256 // keys per attention chunk
#define PW 130 // floats per attention partial: acc[128], max, sum
#define TG uint t [[thread_index_in_threadgroup]], \
  uint sg [[simdgroup_index_in_threadgroup]], \
  uint ln [[thread_index_in_simdgroup]]
#define BAR { threadgroup_barrier(mem_flags::mem_threadgroup); \
  threadgroup_barrier(mem_flags::mem_device); }
#define F(i) as_type<float>(p[i])
typedef const device uint *U;
typedef device float *A;
typedef const device float *S;

// sum of v over the threadgroup
static float red(float v, threadgroup float *r, uint sg, uint ln) {
  float o = simd_sum(v);
  BAR
  if (!ln) r[sg] = o;
  BAR
  o = 0;
  for (uint i = 0; i < T / 32; i++) o += r[i];
  return o;
}

// g[r] = W[r] . x for rows | T, split over T/rows column slices
static void smv(uint rows, uint cols, S W, S x, threadgroup float *part,
                threadgroup float *g, uint t) {
  uint k = T / rows, r = t % rows, c = t / rows * cols / k,
       c1 = (t / rows + 1) * cols / k;
  float o = 0;
  for (; c < c1; c++) o += W[r * cols + c] * x[c];
  BAR
  part[t] = o;
  BAR
  if (t < rows) {
    for (o = 0, c = 0; c < k; c++) o += part[t + c * rows];
    g[t] = o;
  }
  BAR
}

// causal conv with residual over the last 3 inputs st and x; shifts st
static float conv(float x, device float *st, S w, float4 d) {
  float4 v = float4(st[0], st[1], st[2], x);
  st[0] = v.y, st[1] = v.z, st[2] = x;
  bool dyn = any(d != 0);
  return (x + dot(v, float4(w[0], w[1], w[2], w[3])) + dot(v, d)) *
         (dyn ? 1 / 3. : 0.4472135955);
}

// x[j] = norm_j(control_j(mix_j(stack, h))) for j < nx, where stack is the
// nb normalized blocks at bk plus norm(h), stored at bk + nb*N.
// p: 0 h, 1 bk, 2 nb, 3 nq, 4 q, 5 x, 6 nx, 7 msg (0: none), 8 ctl,
// 9 first control, 10.. nx norm weights, 13 tmp
static void mixctl(U p, A a, S s, threadgroup float *r,
                   threadgroup float *part, threadgroup float *g, uint t,
                   uint sg, uint ln) {
  uint nb = p[2], nq = p[3];
  A h = a + p[0], bk = a + p[1], tmp = a + p[13];
  float v[3], m[3], o = 0;
  for (uint i = 0; i < 3; i++) v[i] = h[t + i * T], o += v[i] * v[i];
  o = rsqrt(red(o, r, sg, ln) / N + 1e-6);
  for (uint i = 0; i < 3; i++) bk[nb * N + t + i * T] = v[i] * o;
  BAR
  for (uint j = 0; j < p[6]; j++) {
    for (uint i = 0; i < 3; i++) v[i] = h[t + i * T];
    if (nq) {
      S q = s + p[4] + min(j, nq - 1) * N;
      float sc[5], mx = -1e30, z = 0;
      for (uint b = 0; b <= nb; b++) {
        for (uint i = o = 0; i < 3; i++)
          o += bk[b * N + t + i * T] * q[t + i * T];
        sc[b] = red(o, r, sg, ln) / 48;
        mx = max(mx, sc[b]);
      }
      for (uint b = 0; b <= nb; b++) z += sc[b] = exp(sc[b] - mx);
      for (uint i = 0; i < 3; i++)
        for (uint b = v[i] = 0; b <= nb; b++)
          v[i] += sc[b] / z * bk[b * N + t + i * T];
    }
    if (p[7]) {
      A msg = a + p[7] + (p[9] + j) * N;
      S ct = s + p[8] + (p[9] + j) * 256 * N;
      float c = 0;
      for (uint i = o = 0; i < 3; i++)
        m[i] = msg[t + i * T], o += v[i] * v[i], c += m[i] * m[i];
      o = rsqrt(red(o, r, sg, ln) / N + 1e-6);
      c = rsqrt(red(c, r, sg, ln) / N + 1e-6);
      for (uint i = 0; i < 3; i++)
        tmp[t + i * T] = v[i] * o, tmp[N + t + i * T] = m[i] * c;
      BAR
      smv(64, 2 * N, ct, tmp, part, g, t);
      for (uint i = 0; i < 3; i++) {
        S w = ct + 128 * N + (t + i * T) * 64;
        for (uint k = o = c = 0; k < 64; k++)
          o += w[k] * g[k], c += w[64 * N + k] * g[k];
        v[i] += tanh(o) * m[i] - tanh(c) * v[i];
      }
    }
    for (uint i = o = 0; i < 3; i++) o += v[i] * v[i];
    o = rsqrt(red(o, r, sg, ln) / N + 1e-6);
    for (uint i = 0; i < 3; i++)
      a[p[5] + j * N + t + i * T] = v[i] * o * s[p[10 + j] + t + i * T];
  }
  BAR
}

kernel void prep(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 TG) {
  threadgroup float r[T / 32], part[T], g[64];
  mixctl(p, a, s, r, part, g, t, sg, ln);
}

// after the projections: sink, conv on k and v, clips, rms; stores k, v
// p: 0 x (3 inputs), 1 q, 2 k, 3 v, 4 rel, 5 sink out, 6 sink w,
// 7 q clip, 8 rel clip, 9 k conv, 10 v conv, 11 k dyn (0: none), 12 v dyn,
// 13 conv state, 14 kv write (halfs), 15 tau, 16 qk clip, 17 rel clip
kernel void post(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 device half *kv [[buffer(3)]], A cs [[buffer(4)]], TG) {
  threadgroup float part[T], g[24], dk[32], dv[32];
  smv(24, N, s + p[6], a + p[0], part, g, t);
  if (t < 24) a[p[5] + t] = g[t];
  if (p[11]) {
    smv(32, N, s + p[11], a + p[0] + N, part, dk, t);
    smv(32, N, s + p[12], a + p[0] + 2 * N, part, dv, t);
  }
  for (uint c = t; c < 1024; c += T) {
    float4 d = 0, e = 0;
    if (p[11])
      for (uint i = 0; i < 4; i++)
        d[i] = dk[c / 128 * 4 + 3 - i], e[i] = dv[c / 128 * 4 + 3 - i];
    a[p[2] + c] = conv(a[p[2] + c], cs + p[13] + c * 3, s + p[9] + c * 4, d);
    a[p[3] + c] = conv(a[p[3] + c], cs + p[13] + (1024 + c) * 3,
                       s + p[10] + c * 4, e);
  }
  BAR
  float o = 0;
  float4 x;
  uint i = sg * 128 + ln * 4;
  for (uint k = 0; k < 4; k++) x[k] = a[p[1] + i + k] * s[p[7] + i + k];
  o = F(15) / max(1., (sqrt(simd_sum(dot(x, x))) + 1e-6) / F(16));
  for (uint k = 0; k < 4; k++) a[p[1] + i + k] = x[k] * o;
  i = sg * 16 + ln;
  o = ln < 16 ? a[p[4] + i] * s[p[8] + i] : 0;
  o *= F(15) / max(1., (sqrt(simd_sum(o * o)) + 1e-6) / F(17));
  if (ln < 16) a[p[4] + i] = o;
  if (sg < 8) {
    i = sg * 128 + ln * 4;
    for (uint k = 0; k < 4; k++) x[k] = a[p[2] + i + k];
    x /= max(1., (sqrt(simd_sum(dot(x, x))) + 1e-6) / F(16));
    for (uint k = 0; k < 4; k++) kv[p[14] + i + k] = a[p[2] + i + k] = x[k];
    for (uint k = 0; k < 4; k++) x[k] = a[p[3] + i + k];
    x *= rsqrt(simd_sum(dot(x, x)) / 128 + 1e-6);
    for (uint k = 0; k < 4; k++)
      kv[p[14] + 1024 + i + k] = a[p[3] + i + k] = x[k];
  }
}

// one query over a chunk of C keys per (chunk, kv head) threadgroup of 96:
// partial softmax (acc, max, sum) for the 3 q heads of the kv head
// p: 0 q, 1 rel, 2 partials, 3 kv slot (halfs), 4 keys, 5 pos, 6 swa,
// 7 rel bias
kernel void attn(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 const device half *kv [[buffer(3)]],
                 uint3 gp [[threadgroup_position_in_grid]], TG) {
  threadgroup float sc[3 * C];
  uint h = gp.y * 3 + sg, j0 = gp.x * C, j1 = min(j0 + C, p[4]);
  const device half4 *k = (const device half4 *)(kv + p[3] + gp.y * 128);
  const device float4 *q = (const device float4 *)(a + p[0] + h * 128);
  float m = -1e30, l = 0;
  for (uint j = j0 + ln; j < j1; j += 32) {
    uint d = p[6] ? (p[5] - j) & 1023 : p[5] - j;
    float o = 0;
    for (uint i = 0; i < 32; i++) o += dot(float4(k[j * 512 + i]), q[i]);
    o *= 0.08838834764;
    if (d < 1024)
      for (uint i = 0; i < 16; i++)
        o += s[p[7] + d * 16 + i] * a[p[1] + h * 16 + i];
    m = max(m, sc[sg * C + j - j0] = o);
  }
  m = simd_max(m);
  for (uint j = j0 + ln; j < j1; j += 32)
    sc[sg * C + j - j0] = exp(sc[sg * C + j - j0] - m);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc = 0;
  for (uint j = j0; j < j1; j++) {
    float w = sc[sg * C + j - j0];
    l += w, acc += w * float4(k[j * 512 + 256 + ln]);
  }
  A o = a + p[2] + (gp.x * 24 + h) * PW;
  for (uint i = 0; i < 4; i++) o[ln * 4 + i] = acc[i];
  if (!ln) o[128] = m, o[129] = l;
}

// merge the partials with the sink; gate with sigmoid
// p: 0 partials, 1 chunks, 2 sink, 3 gate, 4 out
kernel void merge(U p [[buffer(0)]], A a [[buffer(1)]], TG) {
  float m = a[p[2] + sg], l = 0;
  float4 acc = 0;
  for (uint c = 0; c < p[1]; c++)
    m = max(m, a[p[0] + (c * 24 + sg) * PW + 128]);
  for (uint c = 0; c < p[1]; c++) {
    S o = a + p[0] + (c * 24 + sg) * PW;
    float w = exp(o[128] - m);
    l += w * o[129];
    acc += w * float4(o[ln * 4], o[ln * 4 + 1], o[ln * 4 + 2], o[ln * 4 + 3]);
  }
  acc /= l + exp(a[p[2] + sg] - m);
  for (uint i = 0, e = sg * 128 + ln * 4; i < 4; i++, e++)
    a[p[4] + e] = acc[i] / (1 + exp(-a[p[3] + e]));
}

// conv on the attention output, residual, then the ffn input and router
// p: 0..13 mixctl (h = ffn_inp), 14 attn out, 15 h (0: block start),
// 16 conv state, 17 conv w, 18 router w, 19 router bias, 20 weights out
kernel void mid(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                A cs [[buffer(4)]], device uint *id [[buffer(5)]], TG) {
  threadgroup float r[T / 32], part[T], g[128], w8[8];
  for (uint e = t; e < N; e += T)
    a[p[0] + e] = conv(a[p[14] + e], cs + p[16] + e * 3, s + p[17] + e * 4,
                       0) + (p[15] ? a[p[15] + e] : 0);
  BAR
  mixctl(p, a, s, r, part, g, t, sg, ln);
  smv(128, N, s + p[18], a + p[5], part, g, t);
  float pr = 0;
  if (t < 128) pr = 1 / (1 + exp(-g[t])), part[t] = pr + s[p[19] + t];
  BAR
  if (t < 128) {
    uint k = 0;
    for (uint i = 0; i < 128; i++)
      k += part[i] > part[t] || (part[i] == part[t] && i < t);
    if (k < 8) id[k] = t, w8[k] = pr;
  }
  BAR
  for (uint i = pr = 0; i < 8; i++) pr += w8[i];
  if (t < 8) a[p[20] + t] = w8[t] / (pr + 1e-20) * 8;
}

// situ on the 8 expert gate_up rows, times router weight and scales
// p: 0 gate_up, 1 out, 2 weights, 3 up scale, 4 down scale
kernel void act(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                const device uint *id [[buffer(5)]],
                uint i [[thread_position_in_grid]]) {
  uint e = i / 1152, c = i % 1152;
  if (e > 7) return;
  float us = s[p[3] + id[e]], g = a[p[0] + e * N + c] * us,
        u = a[p[0] + e * N + 1152 + c] * us;
  a[p[1] + i] = 4 * tanh(g / 4) / (1 + exp(-g)) * 25 * tanh(u / 25) *
                a[p[2] + e] * s[p[4] + id[e]];
}

// conv on the ffn output and residual
// p: 0 ffn out, 1 ffn_inp, 2 out, 3 conv state, 4 conv w
kernel void fin(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                A cs [[buffer(4)]], uint e [[thread_position_in_grid]]) {
  if (e < N)
    a[p[2] + e] = a[p[1] + e] +
                  conv(a[p[0] + e], cs + p[3] + e * 3, s + p[4] + e * 4, 0);
}

// cross-pass messages, one threadgroup per core layer: mix of the bank
// block and the pass output with 4 queries each
// p: 0 bank block (normalized), 1 pass output, 2 queries, 3 messages
kernel void msg(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                uint3 gp [[threadgroup_position_in_grid]], TG) {
  threadgroup float r[T / 32];
  float v[3], b[3], o = 0;
  for (uint i = 0; i < 3; i++)
    v[i] = a[p[1] + t + i * T], b[i] = a[p[0] + t + i * T], o += v[i] * v[i];
  o = rsqrt(red(o, r, sg, ln) / N + 1e-6);
  for (uint j = gp.x * 4; j < gp.x * 4 + 4; j++) {
    float x = 0, y = 0;
    for (uint i = 0; i < 3; i++) {
      float q = s[p[2] + j * N + t + i * T];
      x += b[i] * q, y += v[i] * o * q;
    }
    x = red(x, r, sg, ln), y = red(y, r, sg, ln);
    x = 1 / (1 + exp((y - x) / 48));
    for (uint i = 0; i < 3; i++)
      a[p[3] + j * N + t + i * T] = x * b[i] + (1 - x) * v[i] * o;
  }
}

// greedy token: argmax of p[1] logits at p[0], to id[p[2]]
kernel void amax(U p [[buffer(0)]], A a [[buffer(1)]],
                 device uint *id [[buffer(5)]], TG) {
  threadgroup float bv[T];
  threadgroup uint bi[T];
  float m = -1e30;
  uint k = 0;
  for (uint i = t; i < p[1]; i += T)
    if (a[p[0] + i] > m) m = a[p[0] + i], k = i;
  bv[t] = m, bi[t] = k;
  BAR
  if (t) return;
  for (uint i = 1; i < T; i++)
    if (bv[i] > m || (bv[i] == m && bi[i] < k)) m = bv[i], k = bi[i];
  id[p[2]] = k;
}
