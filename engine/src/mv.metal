// weight mat-vec kernels, one token. a row is cut into units of 8
// elements; d_<fmt>(w, x, u0, u1) is the dot of units [u0, u1) of the
// row at w with x. a threadgroup is one simdgroup: lane l takes an
// equal share of the units of NR rows, and simd_sum joins them.
// (grids.metal is prepended by mv.rs)
#include <metal_stdlib>
using namespace metal;

#ifndef NR
#define NR 2
#endif

typedef device const uchar *W;
typedef device const float *X;
typedef device const ushort *H;
typedef device const packed_uchar4 *Q;

struct P { uint n_in, n_out, rb, ne; };

constant float k_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113};
constant float k_fp4[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6,
                                -0., -.5f, -1, -1.5f, -2, -3, -4, -6};

#define LOOP for (uint u = u0; u < u1; u++)
#define XA float4(((device const packed_float4 *)(x + 8*u))[0])
#define XB float4(((device const packed_float4 *)(x + 8*u))[1])
#define F16(p) float(*(device const half *)(p))
#define DOT8(a, b) (dot(float4(a), XA) + dot(float4(b), XB))

static inline float d_f32(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        device const packed_float4 *a = (device const packed_float4 *)(w + 32*u);
        s += DOT8(a[0], a[1]);
    }
    return s;
}

static inline float d_bf16(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        device const packed_ushort4 *a = (device const packed_ushort4 *)(w + 16*u);
        s += DOT8(as_type<float4>(uint4(ushort4(a[0])) << 16),
                  as_type<float4>(uint4(ushort4(a[1])) << 16));
    }
    return s;
}

static inline float d_q8_0(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/4)*34;
        device const packed_char4 *q = (device const packed_char4 *)(b + 2 + 8*(u%4));
        s += F16(b)*DOT8(char4(q[0]), char4(q[1]));
    }
    return s;
}

// q4_k/q5_k: 6-bit scale and min of sub-block j
static inline float2 k4(W sc, uint j) {
    return j < 4 ? float2(sc[j] & 63, sc[j + 4] & 63)
                 : float2((sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4),
                          (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
}

static inline float d_q4_k(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*144;
        uint e = u%32, j = e/4, sh = 4*(j & 1);
        float2 sm = k4(b + 4, j);
        Q q = (Q)(b + 16 + 32*(j/2) + 8*(e%4));
        float4 a = XA, c = XB;
        s += F16(b)*sm.x*(dot(float4((uint4(q[0]) >> sh) & 15), a) +
                          dot(float4((uint4(q[1]) >> sh) & 15), c)) -
             F16(b + 2)*sm.y*(dot(a, 1) + dot(c, 1));
    }
    return s;
}

static inline float d_q5_k(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*176;
        uint e = u%32, j = e/4, sh = 4*(j & 1), o = 8*(e%4);
        float2 sm = k4(b + 4, j);
        Q h = (Q)(b + 16 + o), q = (Q)(b + 48 + 32*(j/2) + o);
        float4 a = XA, c = XB;
        s += F16(b)*sm.x*(
                 dot(float4(((uint4(q[0]) >> sh) & 15) + (((uint4(h[0]) >> j) & 1) << 4)), a) +
                 dot(float4(((uint4(q[1]) >> sh) & 15) + (((uint4(h[1]) >> j) & 1) << 4)), c)) -
             F16(b + 2)*sm.y*(dot(a, 1) + dot(c, 1));
    }
    return s;
}

static inline float d_q6_k(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*210;
        uint e = u%32, n = e/16, g = (e/4)%4, o = 8*(e%4);
        Q q = (Q)(b + 64*n + 32*(g & 1) + o), h = (Q)(b + 128 + 32*n + o);
        uint sh = 4*(g/2);
        float sc = (float)((device const char *)b)[192 + 8*n + 2*g + o/16];
        int4 v0 = int4(((uint4(q[0]) >> sh) & 15) | (((uint4(h[0]) >> 2*g) & 3) << 4)) - 32;
        int4 v1 = int4(((uint4(q[1]) >> sh) & 15) | (((uint4(h[1]) >> 2*g) & 3) << 4)) - 32;
        s += F16(b + 208)*sc*DOT8(v0, v1);
    }
    return s;
}

#define LUT4(t, v) float4(t[v.x], t[v.y], t[v.z], t[v.w])

static inline float d_iq4_nl(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/4)*18;
        uint k = u%4, sh = 4*(k/2);
        Q q = (Q)(b + 2 + 8*(k & 1));
        uint4 v0 = (uint4(q[0]) >> sh) & 15, v1 = (uint4(q[1]) >> sh) & 15;
        s += F16(b)*DOT8(LUT4(k_iq4nl, v0), LUT4(k_iq4nl, v1));
    }
    return s;
}

// unsigned e4m3 scale: the 7 bits as an f16 are the value * 2^-8
static inline float d_nvfp4(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/8)*36;
        uint k = u%8, sh = 4*(k & 1), c = b[k/2];
        Q q = (Q)(b + 4 + 8*(k/2));
        uint4 v0 = (uint4(q[0]) >> sh) & 15, v1 = (uint4(q[1]) >> sh) & 15;
        float d = c == 127 ? 0 : (float)as_type<half>((ushort)((c & 127) << 7))*256;
        s += d*DOT8(LUT4(k_fp4, v0), LUT4(k_fp4, v1));
    }
    return s;
}

// 8 grid magnitudes with 8 sign bits, times x
#define SGN(v, sg, m) select(float4(v), -float4(v), (uint4(sg) & m) != 0)
#define M0 uint4(1, 2, 4, 8)
#define M1 uint4(16, 32, 64, 128)
#define GDOT(g0, g1, sg) \
    DOT8(SGN(as_type<uchar4>(uint(g0)), sg, M0), SGN(as_type<uchar4>(uint(g1)), sg, M1))

static inline float d_iq2_xxs(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*66;
        uint e = u%32, ib = e/4, l = e%4;
        H a = (H)(b + 2 + 8*ib);
        uint aux = a[2] | (uint(a[3]) << 16);
        ulong g = iq2xxs_grid[b[2 + 8*ib + l]];
        uint sg = ksigns_iq2xs[(aux >> 7*l) & 127];
        s += F16(b)*(0.5f + (aux >> 28))*0.25f*GDOT(g, g >> 32, sg);
    }
    return s;
}

static inline float d_iq2_xs(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*74;
        uint e = u%32, q = ((H)(b + 2))[e];
        ulong g = iq2xs_grid[q & 511];
        uint sg = ksigns_iq2xs[q >> 9];
        float ls = (b[66 + e/4] >> 4*((e%4)/2)) & 15;
        s += F16(b)*(0.5f + ls)*0.25f*GDOT(g, g >> 32, sg);
    }
    return s;
}

static inline float d_iq2_s(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*82;
        uint e = u%32;
        ulong g = iq2s_grid[b[2 + e] | (((b[66 + e/4] >> 2*(e%4)) & 3) << 8)];
        uint sg = b[34 + e];
        float ls = (b[74 + e/4] >> 4*((e%4)/2)) & 15;
        s += F16(b)*(0.5f + ls)*0.25f*GDOT(g, g >> 32, sg);
    }
    return s;
}

static inline float d_iq3_xxs(W w, X x, uint u0, uint u1) {
    float s = 0;
    LOOP {
        W b = w + (u/32)*98;
        uint e = u%32, ib = e/4, l = e%4;
        H a = (H)(b + 66 + 4*ib);
        uint aux = a[0] | (uint(a[1]) << 16);
        uint sg = ksigns_iq2xs[(aux >> 7*l) & 127];
        s += F16(b)*(0.5f + (aux >> 28))*0.5f*
             GDOT(iq3xxs_grid[b[2 + 2*e]], iq3xxs_grid[b[3 + 2*e]], sg);
    }
    return s;
}

#define ARGS constant P &p, W w [[buffer(1)]], X x [[buffer(2)]], \
    device float *y [[buffer(3)]], device const uint *ids [[buffer(4)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    ushort l [[thread_index_in_threadgroup]]
#define LANE uint U = p.n_in/8, u0 = l*U/32, u1 = (l + 1)*U/32

// mv: y[n_out] = W x.  up: y[e][n_out] = W[ids[e]] x, e < ne.
// down: y[n_out] = sum_e W[ids[e]] x[e].  row: y[n_in] = row of W
// (x is an 8x8 identity; thread i takes element i).
#define KERNELS(F) \
kernel void mv_##F(ARGS) { \
    LANE; \
    for (uint i = 0; i < NR; i++) { \
        uint r = tg.x*NR + i; \
        if (r >= p.n_out) break; \
        float s = simd_sum(d_##F(w + (ulong)r*p.rb, x, u0, u1)); \
        if (l == 0) y[r] = s; \
    } \
} \
kernel void up_##F(ARGS) { \
    LANE; \
    for (uint i = 0; i < NR; i++) { \
        uint r = tg.x*NR + i, e = r/p.n_out; \
        if (e >= p.ne) break; \
        ulong o = (ulong)ids[e]*p.n_out + r%p.n_out; \
        float s = simd_sum(d_##F(w + o*p.rb, x, u0, u1)); \
        if (l == 0) y[r] = s; \
    } \
} \
kernel void down_##F(ARGS) { \
    LANE; \
    for (uint i = 0; i < NR; i++) { \
        uint r = tg.x*NR + i; \
        if (r >= p.n_out) break; \
        float s = 0; \
        for (uint e = 0; e < p.ne; e++) \
            s += d_##F(w + ((ulong)ids[e]*p.n_out + r)*p.rb, x + e*p.n_in, u0, u1); \
        s = simd_sum(s); \
        if (l == 0) y[r] = s; \
    } \
} \
kernel void row_##F(constant P &p, W w [[buffer(1)]], X x [[buffer(2)]], \
        device float *y [[buffer(3)]], uint3 t [[thread_position_in_grid]]) { \
    uint u = t.x/8; \
    if (t.x < p.n_in) \
        y[t.x] = d_##F(w, x + 8*(int)(t.x%8) - 8*(int)u, u, u + 1); \
}

KERNELS(f32)
KERNELS(bf16)
KERNELS(q8_0)
KERNELS(q4_k)
KERNELS(q5_k)
KERNELS(q6_k)
KERNELS(iq2_xxs)
KERNELS(iq2_xs)
KERNELS(iq2_s)
KERNELS(iq3_xxs)
KERNELS(iq4_nl)
KERNELS(nvfp4)

// device bandwidth ceiling: every thread sums a contiguous run of words
kernel void bw(constant P &p, device const uint4 *w [[buffer(1)]],
        device uint *y [[buffer(3)]], uint3 t [[thread_position_in_grid]]) {
    uint4 s = 0;
    for (uint i = 0; i < p.n_in; i++) s += w[(ulong)t.x*p.n_in + i];
    y[t.x] = s.x + s.y + s.z + s.w;
}
