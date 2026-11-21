// dspark drafter kernels (DRAFT.md). params are a uint array p; b = f32
// drafter activations, s = small f32 weights, c = f16 context k/v.
// threadgroups are one simdgroup of 32; a row norm is recomputed by every
// threadgroup: lane ln sums elements ln + 32 i.
#include <metal_stdlib>
using namespace metal;
#define N 2304
#define GP uint3 gp [[threadgroup_position_in_grid]], \
  uint ln [[thread_index_in_simdgroup]]
typedef const device uint *U;
typedef device float *A;
typedef const device float *S;
typedef const device ushort *W; // bf16

static float bf(ushort u) { return as_type<float>(uint(u) << 16); }

// a target feature into b; p[2]: scale by the rms of a[p[2]..]. grid 72
// p: 0 src, 1 dst, 2 norm source
kernel void tap(U p [[buffer(0)]], S a [[buffer(1)]], A b [[buffer(2)]],
                GP) {
  float q = 0;
  for (uint e = ln; e < N; e += 32) q += a[p[2] + e] * a[p[2] + e];
  uint e = gp.x * 32 + ln;
  b[p[1] + e] = a[p[0] + e] * (p[2] ? rsqrt(simd_sum(q) / N + 1e-6) : 1);
}

// context of row gp.y: rms(sum_k softmax(fuse[y])[k] f_k) * w_y. grid
// (72, 4). p: 0 f, 1 out, 2 fuse, 3.. w[4]
kernel void fuse(U p [[buffer(0)]], A b [[buffer(1)]], S s [[buffer(2)]],
                 GP) {
  S f = b + p[0], u = s + p[2] + gp.y * 4;
  float4 w = exp(float4(u[0], u[1], u[2], u[3]));
  w /= w.x + w.y + w.z + w.w;
  float q = 0;
  for (uint e = ln; e < N; e += 32) {
    float v = dot(w, float4(f[e], f[N + e], f[2 * N + e], f[3 * N + e]));
    q += v * v;
  }
  uint e = gp.x * 32 + ln;
  float v = dot(w, float4(f[e], f[N + e], f[2 * N + e], f[3 * N + e]));
  b[p[1] + gp.y * N + e] =
    v * rsqrt(simd_sum(q) / N + 1e-6) * s[p[3 + gp.y] + e];
}

// row norm, not in place; rows past 0 take s[p[3]..] if set. grid
// (72, rows). p: 0 src, 1 dst, 2 w, 3 mask
kernel void norm(U p [[buffer(0)]], A b [[buffer(1)]], S s [[buffer(2)]],
                 GP) {
  S x = b + p[0] + gp.y * N;
  float q = 0;
  for (uint e = ln; e < N; e += 32) q += x[e] * x[e];
  uint e = gp.x * 32 + ln;
  b[p[1] + gp.y * N + e] = p[3] && gp.y ? s[p[3] + e] :
    x[e] * rsqrt(simd_sum(q) / N + 1e-6) * s[p[2] + e];
}

// head norm, then neox rope at position p[3] + row, in place. grid
// (heads, rows). p: 0 x, 1 row stride, 2 w, 3 pos
kernel void rope(U p [[buffer(0)]], A b [[buffer(1)]], S s [[buffer(2)]],
                 GP) {
  A x = b + p[0] + gp.y * p[1] + gp.x * 128;
  S w = s + p[2];
  float4 v = float4(x[ln], x[ln + 32], x[ln + 64], x[ln + 96]);
  v *= rsqrt(simd_sum(dot(v, v)) / 128 + 1e-6) *
       float4(w[ln], w[ln + 32], w[ln + 64], w[ln + 96]);
  float2 t = float(p[3] + gp.y) * exp2(float2(ln, ln + 32) * -0.3114307588);
  float2 c = cos(t), n = sin(t);
  x[ln] = v.x * c.x - v.z * n.x, x[ln + 64] = v.x * n.x + v.z * c.x;
  x[ln + 32] = v.y * c.y - v.w * n.y, x[ln + 96] = v.y * n.y + v.w * c.y;
}

// 2048 floats to the f16 cache. grid 64. p: 0 src, 1 dst
kernel void put(U p [[buffer(0)]], A b [[buffer(1)]],
                device half *c [[buffer(3)]], GP) {
  uint e = gp.x * 32 + ln;
  c[p[1] + e] = b[p[0] + e];
}

// block attention: query (head gp.x, row gp.y) over p[3] context keys
// from position p[4] (ring p[5], 0: none) and the 8 block rows. lane ln
// takes keys ln + 32 i with an online softmax; the lanes merge. grid
// (24, 8). p: 0 q, 1 block k/v, 2 cache, 3 nk, 4 pos0, 5 ring, 6 out
kernel void attn(U p [[buffer(0)]], A b [[buffer(1)]],
                 const device half *c [[buffer(3)]], GP) {
  threadgroup float pm[32], pl[32], pa[32 * 128];
  S q = b + p[0] + gp.y * 3072 + gp.x * 128;
  uint h = gp.x / 3 * 128, nk = p[3];
  float m = -1e30, l = 0, acc[128];
  for (uint i = 0; i < 128; i++) acc[i] = 0;
  for (uint j = ln; j < nk + 8; j += 32) {
    float sc = 0, f, w;
    if (j < nk) {
      uint r = p[4] + j;
      const device half *k = c + p[2] + (p[5] ? r % p[5] : r) * 2048 + h;
      for (uint i = 0; i < 128; i++) sc += q[i] * float(k[i]);
      sc *= 0.08838834764;
      f = exp(m - max(m, sc)), w = exp(sc - max(m, sc));
      for (uint i = 0; i < 128; i++)
        acc[i] = acc[i] * f + w * float(k[1024 + i]);
    } else {
      S k = b + p[1] + (j - nk) * 2048 + h;
      for (uint i = 0; i < 128; i++) sc += q[i] * k[i];
      sc *= 0.08838834764;
      f = exp(m - max(m, sc)), w = exp(sc - max(m, sc));
      for (uint i = 0; i < 128; i++) acc[i] = acc[i] * f + w * k[1024 + i];
    }
    l = l * f + w, m = max(m, sc);
  }
  pm[ln] = m, pl[ln] = l;
  for (uint i = 0; i < 128; i++) pa[ln * 128 + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = 0; t < 32; t++) m = max(m, pm[t]);
  l = 0;
  for (uint t = 0; t < 32; t++) l += pl[t] * exp(pm[t] - m);
  for (uint d = ln; d < 128; d += 32) {
    float o = 0;
    for (uint t = 0; t < 32; t++) o += pa[t * 128 + d] * exp(pm[t] - m);
    b[p[6] + gp.y * 3072 + gp.x * 128 + d] = o / l;
  }
}

// g = silu(g) * u. grid (288, rows). p: 0 g, 1 u
kernel void silu(U p [[buffer(0)]], A b [[buffer(1)]], GP) {
  uint e = gp.y * 9216 + gp.x * 32 + ln;
  float g = b[p[0] + e];
  b[p[0] + e] = g / (1 + exp(-g)) * b[p[1] + e];
}

// x += y. grid (72, rows). p: 0 x, 1 y
kernel void add(U p [[buffer(0)]], A b [[buffer(1)]], GP) {
  uint e = gp.y * N + gp.x * 32 + ln;
  b[p[0] + e] += b[p[1] + e];
}

// j = joint_proj [state, markov_w1[prev], cur, f_rnn] + bias; the last
// third (the latent) through tanh. grid 24
// p: 0 state, 1 cur, 2 f_rnn, 3 j, 4 bias, 5 index of prev in id
kernel void joint(U p [[buffer(0)]], A b [[buffer(1)]], S s [[buffer(2)]],
                  U id [[buffer(4)]], W jp [[buffer(5)]],
                  W mk [[buffer(6)]], GP) {
  uint r = gp.x * 32 + ln;
  W w = jp + r * 5120, v = mk + id[p[5]] * 256;
  float o = s[p[4] + r];
  for (uint i = 0; i < 256; i++)
    o += bf(w[i]) * b[p[0] + i] + bf(w[256 + i]) * bf(v[i]);
  for (uint i = 0; i < N; i++)
    o += bf(w[512 + i]) * b[p[1] + i] + bf(w[512 + N + i]) * b[p[2] + i];
  b[p[3] + r] = r < 512 ? o : tanh(o);
}

// logits = base + wb + P latent for vocab row v, P = output hidden_out
// as f16; and the state update s = c + sigmoid(j[..256]) (s - c),
// c = tanh(j[256..512]). grid 6252. p: 0 state, 1 j, 2 base, 3 out, 4 wb
kernel void pl(U p [[buffer(0)]], A b [[buffer(1)]],
               const device half *pp [[buffer(5)]], GP) {
  uint v = gp.x * 32 + ln;
  float o = b[p[2] + v] + b[p[4] + v];
  for (uint k = 0; k < 256; k++)
    o += float(pp[v * 256 + k]) * b[p[1] + 512 + k];
  b[p[3] + v] = o;
  if (v >= 256) return;
  float c = tanh(b[p[1] + 256 + v]);
  b[p[0] + v] = c + (b[p[0] + v] - c) / (1 + exp(-b[p[1] + v]));
}

// columns p[1].. of P from 8 rows of head outputs. grid 6252. p: 0 src
kernel void pset(U p [[buffer(0)]], A b [[buffer(1)]],
                 device half *pp [[buffer(5)]], GP) {
  uint v = gp.x * 32 + ln;
  for (uint t = 0; t < 8; t++)
    pp[v * 256 + p[1] + t] = b[p[0] + t * 201024 + v];
}

// first argmax of b[p[0]..+p[1]] into id[p[2]]. one group of 768
kernel void amax(U p [[buffer(0)]], A b [[buffer(1)]],
                 device uint *id [[buffer(4)]],
                 uint t [[thread_index_in_threadgroup]]) {
  threadgroup float bv[768];
  threadgroup uint bi[768];
  float m = -1e30;
  uint k = 0;
  for (uint i = t; i < p[1]; i += 768)
    if (b[p[0] + i] > m) m = b[p[0] + i], k = i;
  bv[t] = m, bi[t] = k;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t) return;
  for (uint i = 1; i < 768; i++)
    if (bv[i] > m || (bv[i] == m && bi[i] < k)) m = bv[i], k = bi[i];
  id[p[2]] = k;
}
