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

// q5_k mat-vecs: units 2j, 2j + 1, inputs x0..x3, one scale decode for
// both (the decode limits the one-unit kernel)
static inline float w_q5_k(W b, uint j, float4 x0, float4 x1, float4 x2,
        float4 x3) {
    uint sb = j/2, o = 16*(j & 1);
    float2 d = D2*k4(b + 4, sb);
    device const uint4 *qp = (device const uint4 *)(b + 48 + 32*(sb/2) + o);
    uint4 q = (*qp >> 4*(sb & 1)) & 0x0f0f0f0f;
    q |= ((*(device const uint4 *)(b + 16 + o) >> sb) & 0x01010101) << 4;
    return d.x*(dot(float4(as_type<uchar4>(q.x)), x0) +
                dot(float4(as_type<uchar4>(q.y)), x1) +
                dot(float4(as_type<uchar4>(q.z)), x2) +
                dot(float4(as_type<uchar4>(q.w)), x3)) -
           d.y*dot(x0 + x1 + x2 + x3, 1);
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

// 8 grid magnitudes with 8 sign bits, times x 2^-24. as a subnormal half
// a byte b is b 2^-24, exactly, and a sign is bit 15: a word of the grid
// gives the halves of its bytes 0, 2 and 1, 3, and one multiply spreads
// sign bit i to bits 10 + i and 24 + i. the 2^24 is in the scales
static inline float4 hd4(half2 a, half2 b) {
    return float4(a.x, b.x, a.y, b.y);
}
#define HB(g, t, c) \
    as_type<half2>(((g) & 0x00ff00ff) | ((t) << (c) & 0x80008000))
#define H4(g, t, c, d) hd4(HB(g, t, c), HB((g) >> 8, t, d))
#define GDOT(g0, g1, sg) DOT8(H4(uint(g0), (sg)*0x01000400, 5, 4), \
                              H4(uint(g1), (sg)*0x01000400, 1, 0))

// the grids run slowly from the constant cache: G is the table, and the
// mat-vecs below use a copy in threadgroup memory (tgg), with 4
// simdgroups per threadgroup
#define IQ2XXS(G) \
    uint ib = e/4, l = e%4; \
    H a = (H)(b + 2 + 8*ib); \
    uint aux = a[2] | (uint(a[3]) << 16); \
    ulong g = G[b[2 + 8*ib + l]]; \
    uint sg = ksigns_iq2xs[(aux >> 7*l) & 127]; \
    return F16(b)*(0.5f + (aux >> 28))*0x1p22f*GDOT(g, g >> 32, sg);
#define IQ2XS(G) \
    uint q = ((H)(b + 2))[e]; \
    ulong g = G[q & 511]; \
    uint sg = ksigns_iq2xs[q >> 9]; \
    float ls = (b[66 + e/4] >> 4*((e%4)/2)) & 15; \
    return F16(b)*(0.5f + ls)*0x1p22f*GDOT(g, g >> 32, sg);
#define IQ2S(G) \
    ulong g = G[b[2 + e] | (((b[66 + e/4] >> 2*(e%4)) & 3) << 8)]; \
    uint sg = b[34 + e]; \
    float ls = (b[74 + e/4] >> 4*((e%4)/2)) & 15; \
    return F16(b)*(0.5f + ls)*0x1p22f*GDOT(g, g >> 32, sg);
#define IQ3XXS(G) \
    uint ib = e/4, l = e%4; \
    H a = (H)(b + 66 + 4*ib); \
    uint aux = a[0] | (uint(a[1]) << 16), q = ((H)b)[1 + e]; \
    uint sg = ksigns_iq2xs[(aux >> 7*l) & 127]; \
    return F16(b)*(0.5f + (aux >> 28))*0x1p23f* \
           GDOT(G[q & 255], G[q >> 8], sg);
#define GR(F, B, T, ty) UNIT(F) { B(T) } \
    static inline float t_##F(W b, uint e, float4 xa, float4 xb, \
        threadgroup const ty *G) { B(G) }
GR(iq2_xxs, IQ2XXS, iq2xxs_grid, ulong)
GR(iq2_xs, IQ2XS, iq2xs_grid, ulong)
GR(iq2_s, IQ2S, iq2s_grid, ulong)
GR(iq3_xxs, IQ3XXS, iq3xxs_grid, uint)
#define TGG(T, N, ty) threadgroup ty tgg[N]; \
    for (uint i = ti; i < N; i += tn) tgg[i] = T[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup);

// both nibbles of 4 bytes: xa and xb are the inputs of the low and the
// high nibbles, at elements xo_<fmt>(e) and that + NH of the group
#define NIB(F) static inline float n_##F(W b, uint e, float4 xa, float4 xb)

static inline uint xo_iq4_nl(uint e) { return 32*(e/4) + 4*(e%4); }
// mat-vecs: G[byte] is the halves of its low and high nibbles' values
static inline float t_iq4_nl(W b, uint e, float4 xa, float4 xb,
        threadgroup const half2 *G) {
    W c = b + (e/4)*18;
    uint4 q = uint4(*(Q)(c + 2 + 4*(e%4)));
    half2 v0 = G[q.x], v1 = G[q.y], v2 = G[q.z], v3 = G[q.w];
    return F16(c)*(dot(float4(v0.x, v1.x, v2.x, v3.x), xa) +
                   dot(float4(v0.y, v1.y, v2.y, v3.y), xb));
}
#define n_iq4_nl(b, e, xa, xb) t_iq4_nl(b, e, xa, xb, tgg)
#define T4 threadgroup half2 tgg[256]; \
    for (uint i = ti; i < 256; i += tn) \
        tgg[i] = half2(k_iq4nl[i & 15], k_iq4nl[i >> 4]); \
    threadgroup_barrier(mem_flags::mem_threadgroup);

static inline uint xo_nvfp4(uint e) { return 8*(e & ~1u) + 4*(e & 1); }
// an e2m1 nibble as the bits 9-11 and 15 of a half is its value 2^-14
// (the 2^14 is in the scale): F4(q, s) is nibble s/4 of bytes 0 and 2
#define F4(q, s) as_type<half2>((((q) >> (s)) << 9 & 0x0e000e00) | \
                                (((q) >> (s)) << 12 & 0x80008000))
NIB(nvfp4) {
    W a = b + (e/8)*36;
    uint c = a[e%8/2], q = *(device const uint *)(a + 4 + 4*(e%8));
    float d = c == 127 ? 0 :
        (float)as_type<half>((ushort)((c & 127) << 7))*0x1p22f;
    return d*(dot(hd4(F4(q, 0), F4(q, 8)), xa) +
              dot(hd4(F4(q, 4), F4(q, 12)), xb));
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

// as ACC, a lane taking units 2j, 2j + 1 (j = l%16) of group 2k + l/16
#define ACCW(F, R, w0, x) \
    for (uint k = l/16, j = l%16; k < U/32; k += 2) { \
        device const packed_float4 *xp = \
            (device const packed_float4 *)((x) + 256*k + 16*j); \
        float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3]; \
        for (uint i = 0; i < R; i++) \
            s[i] += w_##F((w0) + i*p.rb + k*gb, j, x0, x1, x2, x3); \
    }

#define ARGS constant P &p, W w [[buffer(1)]], X x [[buffer(2)]], \
    device float *y [[buffer(3)]], device const uint *ids [[buffer(4)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    ushort l [[thread_index_in_simdgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort nsg [[simdgroups_per_threadgroup]]
// simdgroup sg of a threadgroup takes R rows from r0; ti of tn threads
#define HEAD(R) uint U = p.n_in/8, gb = p.rb*256/p.n_in, \
    r0 = (tg.x*nsg + sg)*R, ti = 32*sg + l, tn = 32*nsg; \
    float s[R] = {0}
#define TAIL(R) for (uint i = 0; i < R; i++) { \
        float v = simd_sum(s[i]); \
        if (l == 0) y[r0 + i] = v; \
    }

// mv: y[n_out] = W x.  up: y[e][n_out] = W[ids[e]] x, e < ne.
// down: y[n_out] = sum_e W[ids[e]] x[e].  row: y[n_in] = row of W, row
// ids[0] if ne.
// R rows per threadgroup (NR overrides it); n_out is a multiple of R.
// PRE: the format's threadgroup setup, if any
// SLAB b (tests): every row read is one of the first b bytes' rows
#ifdef SLAB
#define WR(r) ((r) % max(1u, SLAB/p.rb))
#else
#define WR(r) (r)
#endif
#define KERNELS(F, R) KERNELS_(F, R, ACC, )
#define KERNELS_(F, R, ACC, PRE) \
kernel void mv_##F(ARGS) { \
    HEAD(R); \
    PRE \
    ACC(F, R, w + (ulong)WR(r0)*p.rb, x) \
    TAIL(R); \
} \
kernel void up_##F(ARGS) { \
    HEAD(R); \
    PRE \
    ACC(F, R, w + WR((ulong)ids[r0/p.n_out]*p.n_out + r0%p.n_out)*p.rb, x) \
    TAIL(R); \
} \
kernel void down_##F(ARGS) { \
    HEAD(R); \
    PRE \
    for (uint j = 0; j < p.ne; j++) \
        ACC(F, R, w + WR((ulong)ids[j]*p.n_out + r0)*p.rb, \
            x + j*p.n_in) \
    TAIL(R); \
} \
kernel void row_##F(constant P &p, W w [[buffer(1)]], \
        device float *y [[buffer(3)]], device const uint *ids [[buffer(4)]], \
        uint t [[thread_position_in_grid]], \
        ushort ti [[thread_index_in_threadgroup]], \
        ushort tn [[threads_per_threadgroup]]) { \
    PRE \
    float4 xa = 0, xb = 0; \
    if (t%8 < 4) xa[t%4] = 1; else xb[t%4] = 1; \
    if (p.ne) w += (ulong)ids[0]*p.rb; \
    if (t < p.n_in) \
        y[t] = u_##F(w + t/256*(p.rb*256/p.n_in), t%256/8, xa, xb); \
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
KERNELS_(q5_k, R(2), ACCW, )
KERNELS(q6_k, R(2))
#define u_iq2_xxs(b, e, xa, xb) t_iq2_xxs(b, e, xa, xb, tgg)
#define u_iq2_xs(b, e, xa, xb) t_iq2_xs(b, e, xa, xb, tgg)
#define u_iq2_s(b, e, xa, xb) t_iq2_s(b, e, xa, xb, tgg)
#define u_iq3_xxs(b, e, xa, xb) t_iq3_xxs(b, e, xa, xb, tgg)
KERNELS_(iq2_xxs, R(8), ACC, TGG(iq2xxs_grid, 256, ulong))
KERNELS_(iq2_xs, R(4), ACC, TGG(iq2xs_grid, 512, ulong))
KERNELS_(iq2_s, R(8), ACC, TGG(iq2s_grid, 1024, ulong))
KERNELS_(iq3_xxs, R(8), ACC, TGG(iq3xxs_grid, 256, uint))
#undef u_iq2_xxs
#undef u_iq2_xs
#undef u_iq2_s
#undef u_iq3_xxs
KERNELS_(iq4_nl, R(2), ACC16, T4)
KERNELS_(nvfp4, R(2), ACC8, )

// device bandwidth ceiling: every thread sums a run of p.n_in words, of
// its own (p.n_out = 0) or a simdgroup's run, lanes adjacent (1)
kernel void bw(constant P &p, device const uint4 *w [[buffer(1)]],
        device uint *y [[buffer(3)]], uint3 t [[thread_position_in_grid]]) {
    uint4 s = 0;
    ulong b = p.n_out ? (ulong)(t.x & ~31u)*p.n_in + t.x%32 : (ulong)t.x*p.n_in;
    for (uint i = 0, d = p.n_out ? 32 : 1; i < p.n_in; i++) s += w[b + i*d];
    y[t.x] = s.x + s.y + s.z + s.w;
}
