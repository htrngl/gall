// weight mat-vec kernels, one token. a row is cut into groups of 256
// elements, a group into 32 units of 8; u_<fmt>(b, e, xa, xb) is the dot
// of unit e of the group at b with the 8 inputs xa, xb. a threadgroup is
// one simdgroup and R rows: lane l takes unit l of every group, so its
// decode offsets are loop invariant and the lanes read adjacent bytes;
// simd_sum joins them. (grids.metal is prepended by mv.rs)
using namespace metal;

typedef device const uchar *W;
typedef device const float *X;
typedef device const ushort *H;
typedef device const packed_uchar4 *Q;

struct P { uint n_in, n_out, rb, ne; };

constant float k_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                              1, 13, 25, 38, 53, 69, 89, 113};
constant float k_fp4[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6,
                            -0., -.5f, -1, -1.5f, -2, -3, -4, -6};

#define UNIT(F) static inline float u_##F(W b, uint e, float4 xa, float4 xb)
#define F16(p) float(*(device const half *)(p))
#define DOT8(a, b) (dot(float4(a), xa) + dot(float4(b), xb))

UNIT(f32) {
    device const packed_float4 *a = (device const packed_float4 *)(b + 32*e);
    return DOT8(a[0], a[1]);
}

UNIT(bf16) {
    device const packed_ushort4 *a = (device const packed_ushort4 *)(b + 16*e);
    return DOT8(as_type<float4>(uint4(ushort4(a[0])) << 16),
                as_type<float4>(uint4(ushort4(a[1])) << 16));
}

UNIT(q8_0) {
    W c = b + (e/4)*34;
    device const packed_char4 *q = (device const packed_char4 *)(c + 2 + 8*(e%4));
    return F16(c)*DOT8(char4(q[0]), char4(q[1]));
}

// q4_k/q5_k: 6-bit scale and min of sub-block j
static inline float2 k4(W sc, uint j) {
    uint a = sc[j & 3], c = sc[4 + (j & 3)], h = sc[8 + (j & 3)];
    return j < 4 ? float2(a & 63, c & 63)
                 : float2((h & 15) | ((a >> 6) << 4), (h >> 4) | ((c >> 6) << 4));
}
#define D2 float2(*(device const packed_half2 *)b)

// the k-quants unpack 4 bytes in one word: as_type<uchar4> of the result
#define U2(p) (*(device const uint2 *)(p))
#define DOTU(v) DOT8(as_type<uchar4>(v.x), as_type<uchar4>(v.y))
#define SX (dot(xa, 1) + dot(xb, 1))

UNIT(q4_k) {
    uint j = e/4;
    float2 d = D2*k4(b + 4, j);
    uint2 q = (U2(b + 16 + 32*(j/2) + 8*(e%4)) >> 4*(j & 1)) & 0x0f0f0f0f;
    return d.x*DOTU(q) - d.y*SX;
}

UNIT(q5_k) {
    uint j = e/4, o = 8*(e%4);
    float2 d = D2*k4(b + 4, j);
    uint2 q = ((U2(b + 48 + 32*(j/2) + o) >> 4*(j & 1)) & 0x0f0f0f0f) |
              (((U2(b + 16 + o) >> j) & 0x01010101) << 4);
    return d.x*DOTU(q) - d.y*SX;
}

// a block is 210 bytes, so its words are read as halves
static inline uint2 h2(W p) {
    uint4 v = uint4(*(device const packed_ushort4 *)p);
    return v.xz | (v.yw << 16);
}

UNIT(q6_k) {
    uint n = e/16, g = (e/4)%4, o = 8*(e%4);
    uint2 q = ((h2(b + 64*n + 32*(g & 1) + o) >> 4*(g/2)) & 0x0f0f0f0f) |
              (((h2(b + 128 + 32*n + o) >> 2*g) & 0x03030303) << 4);
    float sc = (float)((device const char *)b)[192 + 8*n + 2*g + o/16];
    q = (q + 0x60606060) ^ 0x80808080; // each byte - 32
    return F16(b + 208)*sc*DOT8(as_type<char4>(q.x), as_type<char4>(q.y));
}

#define LUT4(t, v) float4(t[v.x], t[v.y], t[v.z], t[v.w])

UNIT(iq4_nl) {
    W c = b + (e/4)*18;
    uint k = e%4, sh = 4*(k/2);
    Q q = (Q)(c + 2 + 8*(k & 1));
    uint4 v0 = (uint4(q[0]) >> sh) & 15, v1 = (uint4(q[1]) >> sh) & 15;
    return F16(c)*DOT8(LUT4(k_iq4nl, v0), LUT4(k_iq4nl, v1));
}

// unsigned e4m3 scale: the 7 bits as an f16 are the value * 2^-8
UNIT(nvfp4) {
    W a = b + (e/8)*36;
    uint k = e%8, sh = 4*(k & 1), c = a[k/2];
    Q q = (Q)(a + 4 + 8*(k/2));
    uint4 v0 = (uint4(q[0]) >> sh) & 15, v1 = (uint4(q[1]) >> sh) & 15;
    float d = c == 127 ? 0 : (float)as_type<half>((ushort)((c & 127) << 7))*256;
    return d*DOT8(LUT4(k_fp4, v0), LUT4(k_fp4, v1));
}

// 8 grid magnitudes with 8 sign bits, times x
#define SGN(v, sg, m) select(float4(v), -float4(v), (uint4(sg) & m) != 0)
#define M0 uint4(1, 2, 4, 8)
#define M1 uint4(16, 32, 64, 128)
#define GDOT(g0, g1, sg) \
    DOT8(SGN(as_type<uchar4>(uint(g0)), sg, M0), SGN(as_type<uchar4>(uint(g1)), sg, M1))

UNIT(iq2_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 2 + 8*ib);
    uint aux = a[2] | (uint(a[3]) << 16);
    ulong g = iq2xxs_grid[b[2 + 8*ib + l]];
    uint sg = ksigns_iq2xs[(aux >> 7*l) & 127];
    return F16(b)*(0.5f + (aux >> 28))*0.25f*GDOT(g, g >> 32, sg);
}

UNIT(iq2_xs) {
    uint q = ((H)(b + 2))[e];
    ulong g = iq2xs_grid[q & 511];
    uint sg = ksigns_iq2xs[q >> 9];
    float ls = (b[66 + e/4] >> 4*((e%4)/2)) & 15;
    return F16(b)*(0.5f + ls)*0.25f*GDOT(g, g >> 32, sg);
}

UNIT(iq2_s) {
    ulong g = iq2s_grid[b[2 + e] | (((b[66 + e/4] >> 2*(e%4)) & 3) << 8)];
    uint sg = b[34 + e];
    float ls = (b[74 + e/4] >> 4*((e%4)/2)) & 15;
    return F16(b)*(0.5f + ls)*0.25f*GDOT(g, g >> 32, sg);
}

UNIT(iq3_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 66 + 4*ib);
    uint aux = a[0] | (uint(a[1]) << 16);
    uint sg = ksigns_iq2xs[(aux >> 7*l) & 127];
    return F16(b)*(0.5f + (aux >> 28))*0.5f*
           GDOT(iq3xxs_grid[b[2 + 2*e]], iq3xxs_grid[b[3 + 2*e]], sg);
}

// both nibbles of 4 bytes: xa and xb are the inputs of the low and the
// high nibbles, at elements xo_<fmt>(e) and that + NH of the group
#define NIB(F) static inline float n_##F(W b, uint e, float4 xa, float4 xb)
#define NDOT(t, q) (dot(LUT4(t, (q & 15)), xa) + dot(LUT4(t, (q >> 4)), xb))

static inline uint xo_iq4_nl(uint e) { return 32*(e/4) + 4*(e%4); }
NIB(iq4_nl) {
    W c = b + (e/4)*18;
    uint4 q = uint4(*(Q)(c + 2 + 4*(e%4)));
    return F16(c)*NDOT(k_iq4nl, q);
}

static inline uint xo_nvfp4(uint e) { return 8*(e & ~1u) + 4*(e & 1); }
NIB(nvfp4) {
    W a = b + (e/8)*36;
    uint c = a[e%8/2];
    uint4 q = uint4(*(Q)(a + 4 + 4*(e%8)));
    float d = c == 127 ? 0 : (float)as_type<half>((ushort)((c & 127) << 7))*256;
    return d*NDOT(k_fp4, q);
}

#define ACCN(F, NH, R, w0, x) \
    for (uint k = 0; 32*k + l < U; k++) { \
        device const packed_float4 *xp = \
            (device const packed_float4 *)((x) + 256*k + xo_##F(l)); \
        float4 xa = xp[0], xb = xp[NH/4]; \
        for (uint i = 0; i < R; i++) \
            s[i] += n_##F((w0) + i*p.rb + k*gb, l, xa, xb); \
    }
#define ACC16(F, R, w0, x) ACCN(F, 16, R, w0, x)
#define ACC8(F, R, w0, x) ACCN(F, 8, R, w0, x)

// s[i] += unit l of every group of row i of R rows at w0, times x
#define ACC(F, R, w0, x) \
    for (uint k = 0; 32*k + l < U; k++) { \
        device const packed_float4 *xp = \
            (device const packed_float4 *)((x) + 256*k + 8*l); \
        float4 xa = xp[0], xb = xp[1]; \
        for (uint i = 0; i < R; i++) \
            s[i] += u_##F((w0) + i*p.rb + k*gb, l, xa, xb); \
    }

#define ARGS constant P &p, W w [[buffer(1)]], X x [[buffer(2)]], \
    device float *y [[buffer(3)]], device const uint *ids [[buffer(4)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    ushort l [[thread_index_in_threadgroup]]
#define HEAD(R) uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*R; \
    float s[R] = {0}
#define TAIL(R) for (uint i = 0; i < R; i++) { \
        float v = simd_sum(s[i]); \
        if (l == 0) y[r0 + i] = v; \
    }

// mv: y[n_out] = W x.  up: y[e][n_out] = W[ids[e]] x, e < ne.
// down: y[n_out] = sum_e W[ids[e]] x[e].  row: y[n_in] = row of W, row
// ids[0] if ne.
// R rows per threadgroup (NR overrides it); n_out is a multiple of R.
#define KERNELS(F, R) KERNELS_(F, R, ACC)
#define KERNELS_(F, R, ACC) \
kernel void mv_##F(ARGS) { \
    HEAD(R); \
    ACC(F, R, w + (ulong)r0*p.rb, x) \
    TAIL(R); \
} \
kernel void up_##F(ARGS) { \
    HEAD(R); \
    ACC(F, R, w + ((ulong)ids[r0/p.n_out]*p.n_out + r0%p.n_out)*p.rb, x) \
    TAIL(R); \
} \
kernel void down_##F(ARGS) { \
    HEAD(R); \
    for (uint j = 0; j < p.ne; j++) \
        ACC(F, R, w + ((ulong)ids[j]*p.n_out + r0)*p.rb, x + j*p.n_in) \
    TAIL(R); \
} \
kernel void row_##F(constant P &p, W w [[buffer(1)]], \
        device float *y [[buffer(3)]], device const uint *ids [[buffer(4)]], \
        uint3 t [[thread_position_in_grid]]) { \
    float4 xa = 0, xb = 0; \
    if (t.x%8 < 4) xa[t.x%4] = 1; else xb[t.x%4] = 1; \
    if (p.ne) w += (ulong)ids[0]*p.rb; \
    if (t.x < p.n_in) \
        y[t.x] = u_##F(w + t.x/256*(p.rb*256/p.n_in), t.x%256/8, xa, xb); \
}

#ifdef NR
#define R(n) NR
#else
#define R(n) n
#endif
KERNELS(f32, R(2))
KERNELS(bf16, R(2))
KERNELS(q8_0, R(4))
KERNELS(q4_k, R(2))
KERNELS(q5_k, R(2))
KERNELS(q6_k, R(2))
KERNELS(iq2_xxs, R(8))
KERNELS(iq2_xs, R(4))
KERNELS(iq2_s, R(8))
KERNELS(iq3_xxs, R(8))
KERNELS_(iq4_nl, R(4), ACC16)
KERNELS_(nvfp4, R(8), ACC8)

// device bandwidth ceiling: every thread sums a contiguous run of words
kernel void bw(constant P &p, device const uint4 *w [[buffer(1)]],
        device uint *y [[buffer(3)]], uint3 t [[thread_position_in_grid]]) {
    uint4 s = 0;
    for (uint i = 0; i < p.n_in; i++) s += w[(ulong)t.x*p.n_in + i];
    y[t.x] = s.x + s.y + s.z + s.w;
}
