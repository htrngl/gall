// prefill: the small ops with a token per threadgroup of one simdgroup.
// lane ln owns elements ln, ln + 32, ..; sums are simd_sum, so there are
// no barriers. a is rebased to the token's region (STR floats). conv taps
// come from the regions of the 3 tokens before, or from the ring cs for
// the first 3; `end` stores the ring when the visit has read it. every
// product with a weight matrix is a mat-mat of pfm.metal between kernels.
// params are a uint array p (floats bit-cast); offsets in floats, 0: none.
#include <metal_stdlib>
using namespace metal;
#define N 2304
#define CH 6656 // conv channels per slot: k, v, attention out, ffn out
#define C 256 // keys per attention chunk
#define TG uint3 gp [[threadgroup_position_in_grid]], \
  uint ln [[thread_index_in_simdgroup]]
#define EL for (uint e = ln; e < N; e += 32)
#define F(i) as_type<float>(p[i])
#define SUM(o, x) { o = 0; EL o += x; o = simd_sum(o); }
#define RMS(o, x) { SUM(o, (x) * (x)) o = rsqrt(o / N + 1e-6); }
typedef const device uint *U;
typedef device float *A;
typedef const device float *S;
typedef const device half4 *KV;

// causal conv with residual of element o of token tk at pos; st is the
// channel in ring row 0
static float cv(S a, uint o, uint tk, uint pos, S st, S w, float4 d,
                float sc) {
  float4 v;
  for (uint i = 0; i < 4; i++)
    v[i] = 3 - i <= tk ? (a - (3 - i) * STR)[o] : st[(pos + i - 3) % 16 * CH];
  return (v.w + dot(v, float4(w[0], w[1], w[2], w[3])) + dot(v, d)) * sc;
}

// v[j] = mix_j(stack, h) for j < nx; the stack is the nb normalized blocks
// at bk and norm(h), stored at bk + nb*N. with a message, [norm(v[j]),
// norm(msg[j])] goes to cat for the control; else x[j] = norm_j(v[j]).
// p: 0 h, 1 bk, 2 nb, 3 nq, 4 q, 5 x, 6 nx, 7 msg, 9 first control,
// 10.. nx norm weights, 13 v, 14 cat
static void mix(U p, A a, S s, uint ln) {
  uint nb = p[2], nq = p[3];
  A h = a + p[0], bk = a + p[1];
  float o, c;
  RMS(o, h[e])
  EL bk[nb * N + e] = h[e] * o;
  for (uint j = 0; j < p[6]; j++) {
    A v = a + p[13] + j * N, ct = a + p[14] + j * 2 * N;
    EL v[e] = h[e];
    if (nq) {
      S q = s + p[4] + min(j, nq - 1) * N;
      float sc[6], mx = -1e30, z = 0;
      for (uint b = 0; b <= nb; b++) {
        SUM(o, bk[b * N + e] * q[e])
        mx = max(mx, sc[b] = o / 48);
      }
      for (uint b = 0; b <= nb; b++) z += sc[b] = exp(sc[b] - mx);
      EL for (uint b = v[e] = 0; b <= nb; b++)
        v[e] += sc[b] / z * bk[b * N + e];
    }
    RMS(o, v[e])
    if (p[7]) {
      S m = a + p[7] + (p[9] + j) * N;
      RMS(c, m[e])
      EL ct[e] = v[e] * o, ct[N + e] = m[e] * c;
    } else
      EL a[p[5] + j * N + e] = v[e] * o * s[p[10 + j] + e];
  }
}

kernel void prep(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 TG) {
  mix(p, a + gp.x * STR, s, ln);
}

// x[j] = norm_j(v[j] + tanh(gain)*msg[j] - tanh(forget)*v[j])
// p: 5..13 as mix, 15 gain then forget, 2N per control
kernel void ctl(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                TG) {
  a += gp.x * STR;
  for (uint j = 0; j < p[6]; j++) {
    A v = a + p[13] + j * N;
    S m = a + p[7] + (p[9] + j) * N, g = a + p[15] + j * 2 * N;
    float o;
    EL v[e] += tanh(g[e]) * m[e] - tanh(g[N + e]) * v[e];
    RMS(o, v[e])
    EL a[p[5] + j * N + e] = v[e] * o * s[p[10 + j] + e];
  }
}

// after the projections: conv on k and v, clips, rms; k, v to the cache
// p: 0 q, 1 k, 2 v, 3 rel, 4 q clip, 5 rel clip, 6 k conv, 7 v conv,
// 8 k taps (0: none), 9 v taps, 10 conv state, 11 pos of token 0, 12 swa,
// 13 qk clip, 14 rel clip, 15 log floor, 16 log alpha, 17 kv slot (halfs)
kernel void post(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 device half *kv [[buffer(3)]], S cs [[buffer(4)]], TG) {
  uint tk = gp.x, pos = p[11] + tk;
  a += tk * STR, cs += p[10], kv += p[17] + (p[12] ? pos % 2048 : pos) * 2048;
  float o, c = p[8] ? 1 / 3. : 0.4472135955,
           tau = p[12] ? 1 : 1 + F(16) * log(max(1., (pos + 1) / F(15)));
  for (uint hd = 0; hd < 8; hd++) {
    float4 k, v, d = 0, f = 0;
    for (uint i = 0; i < 4 && p[8]; i++)
      d[i] = a[p[8] + hd * 4 + 3 - i], f[i] = a[p[9] + hd * 4 + 3 - i];
    for (uint i = 0, e = hd * 128 + ln; i < 4; i++, e += 32) {
      k[i] = cv(a, p[1] + e, tk, pos, cs + e, s + p[6] + e * 4, d, c);
      v[i] = cv(a, p[2] + e, tk, pos, cs + 1024 + e, s + p[7] + e * 4, f, c);
    }
    k /= max(1., (sqrt(simd_sum(dot(k, k))) + 1e-6) / F(13));
    v *= rsqrt(simd_sum(dot(v, v)) / 128 + 1e-6);
    for (uint i = 0, e = hd * 128 + ln; i < 4; i++, e += 32)
      kv[e] = k[i], kv[1024 + e] = v[i];
  }
  for (uint hd = 0; hd < 24; hd++) {
    float4 q;
    for (uint i = 0, e = hd * 128 + ln; i < 4; i++, e += 32)
      q[i] = a[p[0] + e] * s[p[4] + e];
    q *= tau / max(1., (sqrt(simd_sum(dot(q, q))) + 1e-6) / F(13));
    for (uint i = 0, e = hd * 128 + ln; i < 4; i++, e += 32)
      a[p[0] + e] = q[i];
    uint e = hd * 16 + ln % 16;
    o = ln < 16 ? a[p[3] + e] * s[p[5] + e] : 0;
    o *= tau / max(1., (sqrt(simd_sum(o * o)) + 1e-6) / F(14));
    if (ln < 16) a[p[3] + e] = o;
  }
}

// one query per (kv head, token) threadgroup of 96, a q head per
// simdgroup: online softmax over chunks of C keys, bias and sink inside,
// then the sigmoid gate.
// p: 0 q, 1 rel, 2 sink, 3 kv slot (halfs), 4 gate, 5 pos of token 0,
// 6 swa, 7 rel bias, 8 out, 9 first token
kernel void attn(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                 const device half *kv [[buffer(3)]],
                 uint sg [[simdgroup_index_in_threadgroup]], TG) {
  threadgroup float sc[3 * C];
  uint tk = gp.y + p[9], h = gp.x * 3 + sg, pos = p[5] + tk,
       lo = p[6] && pos > 1023 ? pos - 1023 : 0;
  a += tk * STR;
  const device float4 *q = (const device float4 *)(a + p[0] + h * 128);
  float r[16], m = a[p[2] + h], l = 1;
  for (uint i = 0; i < 16; i++) r[i] = a[p[1] + h * 16 + i];
  float4 acc = 0;
#define KEY(j) (KV)(kv + p[3] + (p[6] ? (j) % 2048 : (j)) * 2048) + gp.x * 32
  for (uint j0 = lo; j0 <= pos; j0 += C) {
    uint j1 = min(j0 + C, pos + 1);
    float cm = -1e30;
    for (uint j = j0 + ln; j < j1; j += 32) {
      KV k = KEY(j);
      float o = 0;
      for (uint i = 0; i < 32; i++) o += dot(float4(k[i]), q[i]);
      o *= 0.08838834764;
      S b = s + p[7] + (pos - j) * 16;
      if (pos - j < 1024)
        for (uint i = 0; i < 16; i++) o += b[i] * r[i];
      cm = max(cm, sc[sg * C + j - j0] = o);
    }
    cm = max(m, simd_max(cm));
    acc *= exp(m - cm), l *= exp(m - cm), m = cm;
    for (uint j = j0 + ln; j < j1; j += 32)
      sc[sg * C + j - j0] = exp(sc[sg * C + j - j0] - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = j0; j < j1; j++) {
      float w = sc[sg * C + j - j0];
      l += w, acc += w * float4((KEY(j))[256 + ln]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0, e = h * 128 + ln * 4; i < 4; i++, e++)
    a[p[8] + e] = acc[i] / l / (1 + exp(-a[p[4] + e]));
}

// ffn_inp = conv(attention out) + residual, then its mix
// p: 0..14 mix (h = ffn_inp), 16 attn out, 17 residual (0: block start),
// 18 conv state, 19 conv w, 20 pos of token 0
kernel void mid(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                S cs [[buffer(4)]], TG) {
  uint tk = gp.x;
  a += tk * STR;
  EL a[p[0] + e] = cv(a, p[16] + e, tk, p[20] + tk, cs + p[18] + e,
                      s + p[19] + e * 4, 0, 0.4472135955) +
                   (p[17] ? a[p[17] + e] : 0);
  mix(p, a, s, ln);
}

// router: the 8 best of sigmoid(logits) + bias, weights sigmoid / sum * 8
// p: 0 logits, 1 bias, 2 weights out
kernel void top(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                device uint *id [[buffer(5)]], TG) {
  if (ln) return;
  a += gp.x * STR;
  float sv[128], w[8], z = 0;
  for (uint r = 0; r < 128; r++)
    sv[r] = 1 / (1 + exp(-a[p[0] + r])) + s[p[1] + r];
  for (uint k = 0; k < 8; k++) {
    uint b = 0;
    for (uint r = 1; r < 128; r++)
      if (sv[r] > sv[b]) b = r;
    id[gp.x * 8 + k] = b, z += w[k] = sv[b] - s[p[1] + b], sv[b] = -1e30;
  }
  for (uint k = 0; k < 8; k++) a[p[2] + k] = w[k] / (z + 1e-20) * 8;
}

// situ on the gate_up row of a (token, k) pair, times router weight and
// scales. grid (36, pairs); pair j is col[j] = token | k << 16
// p: 0 gate_up, 1 out (both in pe), 2 weights, 3 up scale, 4 down scale
kernel void act(U p [[buffer(0)]], S a [[buffer(1)]], S s [[buffer(2)]],
                const device uint *id [[buffer(5)]], A pe [[buffer(6)]],
                const device uint *col [[buffer(7)]], TG) {
  uint c = gp.x * 32 + ln, j = gp.y, tk = col[j] & 0xffff, k = col[j] >> 16,
       x = id[tk * 8 + k];
  float us = s[p[3] + x], g = pe[p[0] + j * N + c] * us,
        u = pe[p[0] + j * N + 1152 + c] * us;
  pe[p[1] + j * 1152 + c] = 4 * tanh(g / 4) / (1 + exp(-g)) * 25 *
                            tanh(u / 25) * a[tk * STR + p[2] + k] * s[p[4] + x];
}

// expert output of a token: the sum of its 8 pairs, id[p2..] their places
// p: 0 down out (in pe), 1 out, 2 pair index
kernel void dsum(U p [[buffer(0)]], A a [[buffer(1)]],
                 const device uint *id [[buffer(5)]], S pe [[buffer(6)]], TG) {
  a += gp.x * STR;
  EL for (uint k = a[p[1] + e] = 0; k < 8; k++)
    a[p[1] + e] += pe[p[0] + id[p[2] + gp.x * 8 + k] * N + e];
}

// conv on the ffn output and residual
// p: 0 ffn out, 1 ffn_inp, 2 out, 3 conv state, 4 conv w, 5 pos of token 0
kernel void fin(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                S cs [[buffer(4)]], TG) {
  a += gp.x * STR;
  EL a[p[2] + e] = a[p[1] + e] + cv(a, p[0] + e, gp.x, p[5] + gp.x,
                                    cs + p[3] + e, s + p[4] + e * 4, 0,
                                    0.4472135955);
}

// end of a visit: the conv inputs of the last 16 tokens go to the ring
// p: 0 pos of token 0, 1 tokens, 2 k, 3 v, 4 attn out, 5 ffn out,
// 6 conv state
kernel void end(U p [[buffer(0)]], S a [[buffer(1)]], A cs [[buffer(4)]],
                TG) {
  uint tk = gp.x;
  a += tk * STR, cs += p[6] + (p[0] + tk) % 16 * CH;
  if (tk + 16 >= p[1])
    for (uint c = ln; c < CH; c += 32)
      cs[c] = a[c < 1024 ? p[2] + c : c < 2048 ? p[3] + c - 1024
                : c < 4352 ? p[4] + c - 2048 : p[5] + c - 4352];
}

// cross-pass messages: mix of the bank block and the pass output with the
// 48 queries of the 12 core layers
// p: 0 bank block (normalized), 1 pass output, 2 queries, 3 messages
kernel void msg(U p [[buffer(0)]], A a [[buffer(1)]], S s [[buffer(2)]],
                TG) {
  a += gp.x * STR;
  S b = a + p[0], v = a + p[1];
  float o, x, y;
  RMS(o, v[e])
  for (uint j = 0; j < 48; j++) {
    S q = s + p[2] + j * N;
    SUM(x, b[e] * q[e])
    SUM(y, v[e] * o * q[e])
    x = 1 / (1 + exp((y - x) / 48));
    EL a[p[3] + j * N + e] = x * b[e] + (1 - x) * v[e] * o;
  }
}
