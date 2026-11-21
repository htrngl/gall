// everything that is not a weight mat-vec. params are a uint array p
// (floats bit-cast); a = f32 activations, s = small f32 weights,
// kv = f16 cache, cs = conv state, id = expert ids / argmax.
// a gpu thread is slow in series (~0.1 us per dependent load) and the
// gpu is wide: kernels keep the per-thread chain short. a sum over a
// hidden vector runs on a threadgroup of T = 576 threads, one float4 each.
#include <metal_stdlib>
using namespace metal;
#define N 2304
#define T 576 // N / 4
#define NS 18 // simdgroups in T
#define CH 6656 // conv channels per slot: k, v, attention out, ffn out
#define CR 16 // conv ring: the last 3 inputs, and 12 steps of rollback
#define SW 2048 // sliding window ring: window 1024 and as much rollback
#define C 256 // keys per attention chunk
#define PW 130 // floats per attention partial: acc[128], max, sum
#define GP uint3 gp [[threadgroup_position_in_grid]], \
  uint ln [[thread_index_in_simdgroup]]
#define TG GP, uint t [[thread_index_in_threadgroup]], \
  uint sg [[simdgroup_index_in_threadgroup]]
#define BAR threadgroup_barrier(mem_flags::mem_threadgroup)
#define F(i) as_type<float>(p[i])
// float4 i of a float pointer: load, store
#define L(x, i) float4(((const device packed_float4 *)(x))[i])
#define W(x, i) ((device packed_float4 *)(x))[i]
typedef const device uint *U;
typedef device float *A;
typedef const device float *S;

// sum of v over a threadgroup of T; r is its scratch, one per sum
static float red(float v, threadgroup float *r, uint sg, uint ln) {
  v = simd_sum(v);
  if (!ln) r[sg] = v;
  BAR;
  for (uint i = v = 0; i < NS; i++) v += r[i];
  return v;
}

// causal conv with residual over inputs pos-3..pos of 4 channels; st is
// their place in ring row 0, w their 4 taps each, d the dynamic taps
static float4 conv(float4 x, A st, S w, uint pos, float4 d, bool store) {
  float4x4 m = transpose(float4x4(L(w, 0), L(w, 1), L(w, 2), L(w, 3)));
  float4 o = x * (1 + m[3] + d.w);
  for (uint k = 0; k < 3; k++)
    o += L(st + (pos - 3 + k) % CR * CH, 0) * (m[k] + d[k]);
  if (store) W(st + pos % CR * CH, 0) = x;
  return o * (any(d != 0) ? 1 / 3. : 0.4472135955);
}

// the residual stream entering a mix, elements 4t..4t+3:
// h = conv(a[p0]) + a[p1] (p0 = 0: no conv; p1 = 0: no residual).
// p: 3 conv state, 4 conv w, 5 pos
static float4 hv(U p, A a, S s, A cs, uint t, bool store) {
  float4 h = 0;
  if (p[1]) h = L(a + p[1], t);
  if (p[0])
    h += conv(L(a + p[0], t), cs + p[3] + 4 * t, s + p[4] + 16 * t, p[5],
              float4(0), store);
  return h;
}

// scalars of a mix, one threadgroup each, grid (nb + 2, max(nq, 1)):
// sc[y][x] = stack[x] . q[y], h unnormalized at x = nb; x = nb + 1: h . h
// p: 0..5 hv, 6 bk, 7 nb, 9 q, 10 sc
kernel void dots(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 A cs [[buffer(4)]], TG) {
  threadgroup float r[NS];
  uint nb = p[7], b = gp.x;
  float4 x = hv(p, a, s, cs, t, false), y = L(s + p[9] + gp.y * N, t);
  if (b < nb) x = L(a + p[6] + b * N, t);
  float o = red(dot(x, b > nb ? x : y), r, sg, ln);
  if (!t) a[p[10] + gp.y * 8 + b] = o;
}

// v[j] = mix_j(stack, h) for j < nx: softmax-weighted sum of the nb
// normalized blocks at bk and norm(h), which lands at bk + nb*N; nq = 0:
// v = h. x[j] = v[j] * w[j]; the rms scale of v is applied downstream
// (p17: here, by that of h). stores h and the conv input. grid 18
// p: 0..5 hv, 2 h out (0: none), 6 bk, 7 nb, 8 nq, 10 sc, 11 nx, 12 v,
// 13 x, 14.. w[3], 17 scale x
kernel void elem(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 A cs [[buffer(4)]], GP) {
  uint nb = p[7], nq = p[8], t = gp.x * 32 + ln;
  S sc = a + p[10];
  float4 h = hv(p, a, s, cs, t, true);
  float is = rsqrt(sc[nb + 1] / N + 1e-6);
  if (p[2]) W(a + p[2], t) = h;
  W(a + p[6] + nb * N, t) = h * is;
  for (uint j = 0; j < p[11]; j++) {
    float4 v = h, x = 0;
    float z = 0, mx = -1e30;
    S c = sc + min(j, nq - 1) * 8;
    for (uint b = 0; b <= nb && nq; b++) mx = max(mx, c[b] * (b < nb ? 1 : is));
    for (uint b = 0; b <= nb && nq; b++) {
      float w = exp((c[b] * (b < nb ? 1 : is) - mx) / 48);
      z += w;
      x += w * (b < nb ? L(a + p[6] + b * N, t) : h * is);
    }
    if (nq) v = x / z;
    W(a + p[12] + j * N, t) = v;
    W(a + p[13] + j * N, t) = v * L(s + p[14 + j], t) * (p[17] ? is : 1);
  }
}

// control, low rank, grid (65, controls): g[r] = down[r] . v and
// g[64 + r] = down[r] . msg for r < 64, then g[128..] = v . v, msg . msg
// p: 0 v, 1 msg, 2 ctl, 3 g
kernel void low(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                TG) {
  threadgroup float r[NS], q[NS];
  float4 v = L(a + p[0] + gp.y * N, t), m = L(a + p[1] + gp.y * N, t);
  S w = s + p[2] + gp.y * 256 * N + gp.x * 2 * N;
  bool n = gp.x > 63;
  float x = simd_sum(dot(m, n ? m : L(w, T + t)));
  if (!ln) q[sg] = x;
  x = red(dot(v, n ? v : L(w, t)), r, sg, ln);
  if (t) return;
  A g = a + p[3] + gp.y * 130 + (n ? 128 : gp.x);
  g[0] = x;
  for (uint i = x = 0; i < NS; i++) x += q[i];
  g[n ? 1 : 64] = x;
}

// control, grid (72, controls): v += tanh(gain g) msg - tanh(forget g) v
// with g from the normalized v and msg; x = v * w
// p: 0 v, 1 msg, 2 ctl, 3 g, 4 x, 5.. w
kernel void ctl(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                GP) {
  uint e = gp.x * 32 + ln;
  S g = a + p[3] + gp.y * 130,
    w = s + p[2] + gp.y * 256 * N + 128 * N + e * 64;
  float2 n = rsqrt(float2(g[128], g[129]) / N + 1e-6);
  float o = 0, f = 0;
  for (uint k = 0; k < 16; k++) {
    float4 x = n.x * L(g, k) + n.y * L(g, 16 + k);
    o += dot(x, L(w, k)), f += dot(x, L(w, 16 * N + k));
  }
  A v = a + p[0] + gp.y * N + e;
  *v += tanh(o) * a[p[1] + gp.y * N + e] - tanh(f) * *v;
  a[p[4] + gp.y * N + e] = *v * s[p[5 + gp.y] + e];
}

// y[r] = W[r] . x, one threadgroup per row. p: 0 x, 1 W, 2 y
kernel void rows(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 TG) {
  threadgroup float r[NS];
  float o = red(dot(L(s + p[1] + gp.x * N, t), L(a + p[0], t)), r, sg, ln);
  if (!t) a[p[2] + gp.x] = o;
}

// after the projections, one threadgroup per q head: rms scales of the
// inputs v, sink, clips, conv on k and v (v follows k in a), rms; stores
// k, v. the scales and rows are sums over T, the head work is simdgroup 0
// p: 0 q, 1 k, 2 rel, 3 q clip, 4 rel clip, 5 k conv, 6 v conv,
// 7 k taps, 8 v taps (0: none), 9 conv state, 10 kv write (halfs), 11 tau,
// 12 qk clip, 13 rel clip, 14 pos, 15 v, 16 nx, 17 x, 18 q scale out,
// 19 sink w, 20 sink out
kernel void post(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 device half *kv [[buffer(3)]], A cs [[buffer(4)]], TG) {
  threadgroup float r[12 * NS];
  uint h = gp.x, i = h * 128 + ln * 4, nx = p[16];
  bool dy = p[7] && h < 8;
  float o[12];
  for (uint j = 0; j < 3; j++) {
    float4 v = L(a + p[15] + min(j, nx - 1) * N, t);
    o[j] = dot(v, v);
  }
  o[3] = dot(L(s + p[19] + h * N, t), L(a + p[17], t));
  for (uint k = 0; k < 8 && dy; k++)
    o[4 + k] = dot(L(s + p[7 + k / 4] + (h * 4 + 3 - k % 4) * N, t),
                   L(a + p[17] + (1 + k / 4) * N, t));
  for (uint k = 0; k < (dy ? 12 : 4); k++) {
    o[k] = simd_sum(o[k]);
    if (!ln) r[k * NS + sg] = o[k];
  }
  BAR;
  if (sg) return;
  for (uint k = 0; k < (dy ? 12 : 4); k++)
    for (uint j = o[k] = 0; j < NS; j++) o[k] += r[k * NS + j];
  float3 is = rsqrt(float3(o[0], o[1], o[2]) / N + 1e-6);
  if (!ln) a[p[20] + h] = o[3] * is.x;
  if (!h && !ln) a[p[18]] = is.x;
  float4 x = L(a + p[0] + i, 0) * L(s + p[3] + i, 0) * is.x;
  x *= F(11) / max(1., (sqrt(simd_sum(dot(x, x))) + 1e-6) / F(12));
  W(a + p[0] + i, 0) = x;
  float y = ln < 16 ? a[p[2] + h * 16 + ln] * s[p[4] + h * 16 + ln] : 0;
  y *= is.x;
  y *= F(11) / max(1., (sqrt(simd_sum(y * y)) + 1e-6) / F(13));
  if (ln < 16) a[p[2] + h * 16 + ln] = y;
  if (h > 7) return;
  for (uint j = 0; j < 2; j++) {
    float4 d = 0;
    if (dy) d = float4(o[4 + j * 4], o[5 + j * 4], o[6 + j * 4],
                       o[7 + j * 4]) * is[1 + j];
    uint c = i + j * 1024;
    x = conv(L(a + p[1] + c, 0) * is[1 + j], cs + p[9] + c,
             s + p[5 + j] + i * 4, p[14], d, true);
    y = simd_sum(dot(x, x));
    x *= j ? rsqrt(y / 128 + 1e-6) : 1 / max(1., (sqrt(y) + 1e-6) / F(12));
    W(a + p[1] + c, 0) = x;
    ((device half4 *)(kv + p[10] + c))[0] = half4(x);
  }
}

// one query over a chunk of C keys per (chunk, kv head) threadgroup of 8
// simdgroups: a simdgroup takes 32 keys in turn with lane = 4 dims, so a
// score is one simd_sum; online softmax (acc, max, sum) for the 3 q heads
// of the kv head, merged over the threadgroup.
// key j is position j, or pos - j in the sliding window ring
// p: 0 q, 1 rel, 2 partials, 3 kv slot (halfs), 4 keys, 5 pos, 6 swa,
// 7 rel bias
kernel void attn(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 const device half *kv [[buffer(3)]], TG) {
  threadgroup float ml[8 * 6];
  threadgroup float4 ac[8 * 3 * 32];
  uint h = gp.y * 3;
  const device half4 *k = (const device half4 *)(kv + p[3] + gp.y * 128);
  float4 q[3], acc[3];
  float3 rl = 0, m = -1e30, l = 0;
  for (uint i = 0; i < 3; i++) {
    q[i] = L(a + p[0] + (h + i) * 128, ln), acc[i] = 0;
    if (ln < 16) rl[i] = a[p[1] + (h + i) * 16 + ln];
  }
  for (uint j = gp.x * C + sg * 32, j1 = min(j + 32, p[4]); j < j1; j++) {
    uint d = p[6] ? j : p[5] - j, r = (p[6] ? (p[5] - j) % SW : j) * 512;
    float4 kk = float4(k[r + ln]), vv = float4(k[r + 256 + ln]);
    float b = 0;
    if (d < 1024 && ln < 16) b = s[p[7] + d * 16 + ln];
    for (uint i = 0; i < 3; i++) {
      float o = simd_sum(dot(kk, q[i]) * 0.08838834764 + b * rl[i]);
      float w = exp(min(o, m[i]) - max(o, m[i]));
      bool up = o > m[i];
      m[i] = max(o, m[i]);
      l[i] = up ? l[i] * w + 1 : l[i] + w;
      acc[i] = up ? acc[i] * w + vv : acc[i] + w * vv;
    }
  }
  for (uint i = 0; i < 3; i++) {
    ac[(sg * 3 + i) * 32 + ln] = acc[i];
    if (!ln) ml[sg * 6 + i] = m[i], ml[sg * 6 + 3 + i] = l[i];
  }
  BAR;
  if (sg > 2) return;
  float mm = -1e30, ll = 0;
  float4 o = 0;
  for (uint g = 0; g < 8; g++) mm = max(mm, ml[g * 6 + sg]);
  for (uint g = 0; g < 8; g++) {
    float w = exp(ml[g * 6 + sg] - mm);
    ll += w * ml[g * 6 + 3 + sg], o += w * ac[(g * 3 + sg) * 32 + ln];
  }
  A out = a + p[2] + (gp.x * 24 + h + sg) * PW;
  W(out, ln) = o;
  if (!ln) out[128] = mm, out[129] = ll;
}

// merge the partials with the sink; gate with sigmoid. one group per head
// p: 0 partials, 1 chunks, 2 sink, 3 gate, 4 out, 5 q scale
kernel void merge(U p [[buffer(0)]], A a [[buffer(1)]], GP) {
  float m = a[p[2] + gp.x], l = 0;
  float4 acc = 0;
  for (uint c = 0; c < p[1]; c++)
    m = max(m, a[p[0] + (c * 24 + gp.x) * PW + 128]);
  for (uint c = 0; c < p[1]; c++) {
    S o = a + p[0] + (c * 24 + gp.x) * PW;
    float w = exp(o[128] - m);
    l += w * o[129], acc += w * L(o, ln);
  }
  acc /= l + exp(a[p[2] + gp.x] - m);
  uint e = gp.x * 128 + ln * 4;
  W(a + p[4] + e, 0) = acc / (1 + exp(-L(a + p[3] + e, 0) * a[p[5]]));
}

// router: rms scale of the ffn input v, then top 8 of sigmoid(logits) +
// bias, weights normalized, times 8. one threadgroup of T
// p: 0 logits, 1 bias, 2 weights out, 3 v, 4 scale out
kernel void top8(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 device uint *id [[buffer(5)]], TG) {
  threadgroup float r[NS], sel[128], w8[8];
  float4 v = L(a + p[3], t);
  float z = rsqrt(red(dot(v, v), r, sg, ln) / N + 1e-6), pr = 0;
  if (!t) a[p[4]] = z;
  if (t < 128) {
    pr = 1 / (1 + exp(-a[p[0] + t] * z));
    sel[t] = pr + s[p[1] + t];
  }
  BAR;
  uint k = 0;
  for (uint i = 0; i < 128 && t < 128; i++)
    k += sel[i] > sel[t] || (sel[i] == sel[t] && i < t);
  if (k < 8 && t < 128) id[k] = t, w8[k] = pr;
  BAR;
  for (uint i = z = 0; i < 8; i++) z += w8[i];
  if (t < 8) a[p[2] + t] = w8[t] / (z + 1e-20) * 8;
}

// situ on the 8 expert gate_up rows, times router weight and scales
// p: 0 gate_up, 1 out, 2 weights, 3 up scale, 4 down scale, 5 input scale
kernel void act(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                const device uint *id [[buffer(5)]],
                uint i [[thread_position_in_grid]]) {
  uint e = i / 1152, c = i % 1152;
  if (e > 7) return;
  float us = s[p[3] + id[e]] * a[p[5]], g = a[p[0] + e * N + c] * us,
        u = a[p[0] + e * N + 1152 + c] * us;
  a[p[1] + i] = 4 * tanh(g / 4) / (1 + exp(-g)) * 25 * tanh(u / 25) *
                a[p[2] + e] * s[p[4] + id[e]];
}

// cross-pass messages, grid (72, 12 core layers): mix of the bank block
// and the pass output with 4 queries each
// p: 0 bank block (normalized), 1 pass output, 2 queries, 3 messages
kernel void msg(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                GP) {
  S b = a + p[0], v = a + p[1];
  uint e0 = gp.x * 32 + ln;
  float o = 0;
  for (uint e = ln; e < N; e += 32) o += v[e] * v[e];
  o = rsqrt(simd_sum(o) / N + 1e-6);
  for (uint j = gp.y * 4; j < gp.y * 4 + 4; j++) {
    S q = s + p[2] + j * N;
    float x = 0, y = 0;
    for (uint e = ln; e < N; e += 32) x += b[e] * q[e], y += v[e] * q[e];
    x = 1 / (1 + exp((simd_sum(y) * o - simd_sum(x)) / 48));
    a[p[3] + j * N + e0] = x * b[e0] + (1 - x) * v[e0] * o;
  }
}

// greedy token, stage 1: (max, index) of each 128 logits, one simdgroup
// p: 0 logits, 1 count, 2 out
kernel void amax(U p [[buffer(0)]], A a [[buffer(1)]], GP) {
  uint i = (gp.x * 32 + ln) * 4, k = ~0u;
  float4 v = L(a + p[0] + i, 0);
  float m = -1e30;
  for (uint c = 0; c < 4; c++)
    if (i + c < p[1] && v[c] > m) m = v[c], k = i + c;
  float b = simd_max(m);
  k = simd_min(m == b ? k : ~0u);
  if (!ln) a[p[2] + gp.x * 2] = b, a[p[2] + gp.x * 2 + 1] = as_type<float>(k);
}

// stage 2: the best of p[1] (max, index) pairs at p[0], to id[p[2]].
// one threadgroup of 1024
kernel void best(U p [[buffer(0)]], A a [[buffer(1)]],
                 device uint *id [[buffer(5)]], TG) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  float m = -1e30;
  uint k = ~0u;
  for (uint i = t; i < p[1]; i += 1024) {
    float v = a[p[0] + i * 2];
    uint j = as_type<uint>(a[p[0] + i * 2 + 1]);
    if (v > m || (v == m && j < k)) m = v, k = j;
  }
  float b = simd_max(m);
  k = simd_min(m == b ? k : ~0u);
  if (!ln) bv[sg] = b, bi[sg] = k;
  BAR;
  if (t) return;
  for (uint i = 0; i < 32; i++)
    if (bv[i] > b || (bv[i] == b && bi[i] < k)) b = bv[i], k = bi[i];
  id[p[2]] = k;
}
