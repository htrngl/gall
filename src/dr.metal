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
#define P4(x) ((const device packed_float4 *)(x))

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

// block attention in two steps, so that no thread runs a long chain and
// the 3 q heads of a kv head share its keys. 1: kv head gp.x, row gp.y
// and chunk gp.z of 256 keys: the p[3] context keys from position p[4]
// (ring p[5], 0: none), then the 8 block rows. each of 16 simdgroups
// takes 16 keys in turn, a lane 4 dims, with an online softmax; they
// merge to (acc[128], max, sum) per q head. grid (8, 8, chunks) of 512
// p: 0 q, 1 block k/v, 2 cache, 3 nk, 4 pos0, 5 ring, 6 partials, 7 chunks
kernel void attn(U p [[buffer(0)]], A b [[buffer(1)]],
                 const device half *c [[buffer(3)]], GP,
                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float4 ac[16 * 3 * 32];
  threadgroup float ms[16 * 3], ls[16 * 3];
  uint h = gp.x * 128, nk = p[3];
  float4 q[3], acc[3] = {0, 0, 0};
  float3 m = -1e30, l = 0, s;
  for (uint i = 0; i < 3; i++)
    q[i] = P4(b + p[0] + gp.y * 3072 + (gp.x * 3 + i) * 128)[ln];
  for (uint j = gp.z * 256 + sg * 16, j1 = min(j + 16, nk + 8); j < j1; j++) {
    float4 k, v;
    if (j < nk) {
      uint r = p[4] + j;
      const device half4 *x = (const device half4 *)
        (c + p[2] + (p[5] ? r % p[5] : r) * 2048 + h);
      k = float4(x[ln]), v = float4(x[256 + ln]);
    } else {
      S x = b + p[1] + (j - nk) * 2048 + h;
      k = P4(x)[ln], v = P4(x)[256 + ln];
    }
    for (uint i = 0; i < 3; i++)
      s[i] = simd_sum(dot(k, q[i])) * 0.08838834764;
    float3 mx = max(m, s), f = exp(m - mx), w = exp(s - mx);
    for (uint i = 0; i < 3; i++) acc[i] = acc[i] * f[i] + w[i] * v;
    l = l * f + w, m = mx;
  }
  for (uint i = 0; i < 3; i++) {
    ac[(sg * 3 + i) * 32 + ln] = acc[i];
    if (!ln) ms[sg * 3 + i] = m[i], ls[sg * 3 + i] = l[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg > 2) return;
  float4 o = 0;
  float mm = -1e30, ll = 0;
  for (uint g = 0; g < 16; g++) mm = max(mm, ms[g * 3 + sg]);
  for (uint g = 0; g < 16; g++) {
    float w = exp(ms[g * 3 + sg] - mm);
    ll += w * ls[g * 3 + sg], o += w * ac[(g * 3 + sg) * 32 + ln];
  }
  A out = b + p[6] + ((gp.y * 24 + gp.x * 3 + sg) * p[7] + gp.z) * 130;
  ((device packed_float4 *)out)[ln] = o;
  if (!ln) out[128] = mm, out[129] = ll;
}

// 2: the chunks of q head gp.x, row gp.y into its output. grid (24, 8)
// p: 0 partials, 1 chunks, 2 out
kernel void amrg(U p [[buffer(0)]], A b [[buffer(1)]], GP) {
  S o = b + p[0] + (gp.y * 24 + gp.x) * p[1] * 130;
  float m = -1e30, l = 0;
  float4 acc = 0;
  for (uint i = 0; i < p[1]; i++) m = max(m, o[i * 130 + 128]);
  for (uint i = 0; i < p[1]; i++) {
    float w = exp(o[i * 130 + 128] - m);
    l += w * o[i * 130 + 129], acc += w * P4(o + i * 130)[ln];
  }
  ((device packed_float4 *)(b + p[2] + gp.y * 3072 + gp.x * 128))[ln] =
    acc / l;
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
// third (the latent) through tanh. grid 768: a row a threadgroup
// p: 0 state, 1 cur, 2 f_rnn, 3 j, 4 bias, 5 index of prev in id
kernel void joint(U p [[buffer(0)]], A b [[buffer(1)]], S s [[buffer(2)]],
                  U id [[buffer(4)]], W jp [[buffer(5)]],
                  W mk [[buffer(6)]], GP) {
  uint r = gp.x;
  W w = jp + r * 5120, v = mk + id[p[5]] * 256;
  float o = 0;
  for (uint i = ln; i < 256; i += 32)
    o += bf(w[i]) * b[p[0] + i] + bf(w[256 + i]) * bf(v[i]);
  for (uint i = ln; i < N; i += 32)
    o += bf(w[512 + i]) * b[p[1] + i] + bf(w[512 + N + i]) * b[p[2] + i];
  o = simd_sum(o) + s[p[4] + r];
  if (!ln) b[p[3] + r] = r < 512 ? o : tanh(o);
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

// the first argmax k of b[p[0]..+p[1]]: id[p[2]] = sl[k], and its
// probability after the softcap p[4] to b[p[3]]. one group of 768
kernel void amax(U p [[buffer(0)]], A b [[buffer(1)]],
                 device uint *id [[buffer(4)]], U sl [[buffer(5)]],
                 uint t [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint ln [[thread_index_in_simdgroup]]) {
  threadgroup float bv[24];
  threadgroup uint bi[24];
  S x = b + p[0];
  float m = -1e30, c = as_type<float>(p[4]), o = 0;
  uint k = 0;
  for (uint i = t; i < p[1]; i += 768)
    if (x[i] > m) m = x[i], k = i;
  for (uint d = 16; d; d /= 2) {
    float n = simd_shuffle_down(m, d);
    uint j = simd_shuffle_down(k, d);
    if (n > m || (n == m && j < k)) m = n, k = j;
  }
  if (!ln) bv[sg] = m, bi[sg] = k;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = bv[0], k = bi[0];
  for (uint i = 1; i < 24; i++)
    if (bv[i] > m || (bv[i] == m && bi[i] < k)) m = bv[i], k = bi[i];
  m = c * tanh(m / c);
  for (uint i = t; i < p[1]; i += 768) o += exp(c * tanh(x[i] / c) - m);
  o = simd_sum(o);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!ln) bv[sg] = o;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t) return;
  for (uint i = 1; i < 24; i++) o += bv[i];
  id[p[2]] = sl[k], b[p[3]] = 1 / o;
}

// the expert slabs of a pass's rows by class, for drm.metal x_*: uses 1,
// 2, 3 or more. entry j of class k (65 entries of 10 per class) is (id, c
// uses, use i: row t << 3 | k), in order of first use. ind: the grids of
// class k at 2, 4, 8, 16 weight rows per threadgroup. one group of 64:
// thread e is use k = e % 8 of row e / 8. p: 0 rows
kernel void uni(U p [[buffer(0)]], U id [[buffer(1)]],
                device uint *tab [[buffer(2)]],
                device uint *ind [[buffer(3)]],
                uint e [[thread_index_in_threadgroup]]) {
  threadgroup uint fk[64];
  uint n = 8 * p[0], x = id[16 * (e / 8) + e % 8], f = e, c = 0, r = 0, j = 0;
  for (uint i = 0; i < n; i++) {
    bool s = id[16 * (i / 8) + i % 8] == x;
    c += s, r += s && i < e, f = s ? min(f, i) : f;
  }
  uint k = min(c, 3u); // the class of a first use, else 0
  fk[e] = e < n && f == e ? k : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < f; i++) j += fk[i] == k;
  if (e < n) {
    device uint *o = tab + (65 * (k - 1) + j) * 10;
    if (f == e) o[0] = x, o[1] = c;
    o[2 + r] = e / 8 << 3 | e % 8;
  }
  if (e >= 3) return;
  for (uint i = j = 0; i < 64; i++) j += fk[i] == e + 1;
  tab[(65 * e + j) * 10 + 1] = 0; // the end, as a grid is not empty
  for (uint r = 0; r < 4; r++) {
    device uint *o = ind + (e * 4 + r) * 3;
    o[0] = 1152 >> r, o[1] = max(j, 1u), o[2] = 1;
  }
}

// d = sum of the 8 expert outputs at g, per row. grid (72, rows)
// p: 0 g, 1 d, 2 row stride
kernel void dsum(U p [[buffer(0)]], A a [[buffer(1)]], GP) {
  uint e = gp.x * 32 + ln;
  A g = a + gp.y * p[2] + p[0] + e;
  float o = 0;
  for (uint k = 0; k < 8; k++) o += g[k * N];
  a[gp.y * p[2] + p[1] + e] = o;
}
