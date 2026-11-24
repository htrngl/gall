// d_<fmt>: the 8 weights of unit e of the group at b, dequantized once,
// and their common scale; their inputs are the float4s at xo_<fmt>(e) and
// NH_<fmt> after it
#define DQ(F) static inline float d_##F(W b, uint e, thread float4 &va, \
                                        thread float4 &vb)
#define XO(F) static inline uint xo_##F(uint e) { return 8*e; }
#define NH_q8_0 4
#define NH_q4_k 4
#define NH_q5_k 4
#define NH_q6_k 4
#define NH_iq2_xxs 4
#define NH_iq2_xs 4
#define NH_iq2_s 4
#define NH_iq3_xxs 4
#define NH_iq4_nl 16
#define NH_nvfp4 8
XO(q8_0) XO(q4_k) XO(q5_k) XO(q6_k) XO(iq2_xxs) XO(iq2_xs) XO(iq2_s)
XO(iq3_xxs)

DQ(q8_0) {
    W c = b + (e/4)*34;
    device const packed_char4 *q =
        (device const packed_char4 *)(c + 2 + 8*(e%4));
    va = float4(char4(q[0])), vb = float4(char4(q[1]));
    return F16(c);
}

DQ(q4_k) {
    uint j = e/4, sh = 4*(j & 1);
    float2 d = D2*k4(b + 4, j);
    Q q = (Q)(b + 16 + 32*(j/2) + 8*(e%4));
    va = d.x*float4((uint4(q[0]) >> sh) & 15) - d.y;
    vb = d.x*float4((uint4(q[1]) >> sh) & 15) - d.y;
    return 1;
}

DQ(q5_k) {
    uint j = e/4, sh = 4*(j & 1), o = 8*(e%4);
    float2 d = D2*k4(b + 4, j);
    Q h = (Q)(b + 16 + o), q = (Q)(b + 48 + 32*(j/2) + o);
    va = d.x*float4(((uint4(q[0]) >> sh) & 15) +
                    (((uint4(h[0]) >> j) & 1) << 4)) - d.y;
    vb = d.x*float4(((uint4(q[1]) >> sh) & 15) +
                    (((uint4(h[1]) >> j) & 1) << 4)) - d.y;
    return 1;
}

DQ(q6_k) {
    uint n = e/16, g = (e/4)%4, o = 8*(e%4), sh = 4*(g/2);
    Q q = (Q)(b + 64*n + 32*(g & 1) + o), h = (Q)(b + 128 + 32*n + o);
    va = float4(int4(((uint4(q[0]) >> sh) & 15) |
                     (((uint4(h[0]) >> 2*g) & 3) << 4)) - 32);
    vb = float4(int4(((uint4(q[1]) >> sh) & 15) |
                     (((uint4(h[1]) >> 2*g) & 3) << 4)) - 32);
    return F16(b + 208)*
        (float)((device const char *)b)[192 + 8*n + 2*g + o/16];
}

DQ(iq4_nl) {
    W c = b + (e/4)*18;
    uint4 q = uint4(*(Q)(c + 2 + 4*(e%4)));
    va = LUT4(k_iq4nl, (q & 15)), vb = LUT4(k_iq4nl, (q >> 4));
    return F16(c);
}

DQ(nvfp4) {
    W a = b + (e/8)*36;
    uint c = a[e%8/2];
    uint4 q = uint4(*(Q)(a + 4 + 4*(e%8)));
    va = LUT4(k_fp4, (q & 15)), vb = LUT4(k_fp4, (q >> 4));
    return c == 127 ? 0 : (float)as_type<half>((ushort)((c & 127) << 7))*256;
}

#define GRID(d, g0, g1, sg) \
    va = SGN(as_type<uchar4>(uint(g0)), sg, M0); \
    vb = SGN(as_type<uchar4>(uint(g1)), sg, M1); \
    return d

DQ(iq2_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 2 + 8*ib);
    uint aux = a[2] | (uint(a[3]) << 16);
    ulong g = iq2xxs_grid[b[2 + 8*ib + l]];
    GRID(F16(b)*(0.5f + (aux >> 28))*0.25f, g, g >> 32,
         ksigns_iq2xs[(aux >> 7*l) & 127]);
}

DQ(iq2_xs) {
    uint q = ((H)(b + 2))[e];
    ulong g = iq2xs_grid[q & 511];
    float ls = (b[66 + e/4] >> 4*((e%4)/2)) & 15;
    GRID(F16(b)*(0.5f + ls)*0.25f, g, g >> 32, ksigns_iq2xs[q >> 9]);
}

DQ(iq2_s) {
    ulong g = iq2s_grid[b[2 + e] | (((b[66 + e/4] >> 2*(e%4)) & 3) << 8)];
    float ls = (b[74 + e/4] >> 4*((e%4)/2)) & 15;
    GRID(F16(b)*(0.5f + ls)*0.25f, g, g >> 32, b[34 + e]);
}

DQ(iq3_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 66 + 4*ib);
    uint aux = a[0] | (uint(a[1]) << 16);
    GRID(F16(b)*(0.5f + (aux >> 28))*0.5f, iq3xxs_grid[b[2 + 2*e]],
         iq3xxs_grid[b[3 + 2*e]], ksigns_iq2xs[(aux >> 7*l) & 127]);
}

// up to 8 activation rows against one read of an expert slab (or of a
// dense matrix: slab 0). entry tg.y of class k in tab (dr.metal uni: 65
// entries of 10 per class) is (slab, uses c, use j: row t << 3 | k):
// y[t][k][r] = W[slab][r] x[t][k]. x rows are xs apart and x[t][k] is kx
// floats after x[t][0]; y likewise with ys and n_out
struct PX { uint n_in, n_out, rb, xs, ys, kx; };
typedef device const packed_float4 *X4;
#define XU(e) (x + ((e) >> 3)*p.xs + ((e) & 7)*p.kx)
#define YU(e) (y + ((e) >> 3)*p.ys + ((e) & 7)*p.n_out)
#define ARG constant PX &p, W w [[buffer(1)]], X x [[buffer(2)]], \
    device float *y [[buffer(3)]], device const uint *tab [[buffer(4)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    ushort l [[thread_index_in_simdgroup]]

// classes 1 and 2: C uses as dots, R rows per threadgroup, lane l unit l
// of each group as in mv.metal; a slab costs a dot per use and weight
#define XI(F, R, C) \
kernel void x_##F##_##C(ARG) { \
    device const uint *u = tab + (65*(C - 1) + tg.y)*10; \
    if (!u[1]) return; \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*R; \
    W w0 = w + ((ulong)u[0]*p.n_out + r0)*p.rb; \
    X4 xp[C]; \
    for (uint j = 0; j < C; j++) xp[j] = (X4)(XU(u[2 + j]) + xo_##F(l)); \
    float s[R][C] = {}; \
    for (uint k = 0; 32*k + l < U; k++) { \
        float4 xa[C], xb[C]; \
        for (uint j = 0; j < C; j++) \
            xa[j] = xp[j][64*k], xb[j] = xp[j][64*k + NH_##F/4]; \
        for (uint i = 0; i < R; i++) { \
            float4 a, b; \
            float d = d_##F(w0 + i*p.rb + k*gb, l, a, b); \
            for (uint j = 0; j < C; j++) \
                s[i][j] += d*(dot(a, xa[j]) + dot(b, xb[j])); \
        } \
    } \
    for (uint i = 0; i < R*C; i++) { \
        float v = simd_sum(s[i/C][i%C]); \
        if (l == 0) YU(u[2 + i%C])[r0 + i/C] = v; \
    } \
}

// class 3, 3 uses or more, on 8x8 simdgroup matrices: a dot costs 8
// multiplies a weight, a tile 8 whatever its uses. a threadgroup takes 16
// rows, two tiles. per 32 inputs lane (m, n) decodes unit 4q + n/2 of row
// m once: its weights i and 4 + i are row m, columns n and n + 1 of tile
// i. the inputs follow the same order: lane (m, n) holds element i of unit
// 4q + m/2 (half m%2) of uses n and n + 1. four sums a tile, as the
// lanes of a dot: shorter chains, and nearer to it
#define TE(a) (*(thread float2 *)&(a).thread_elements())
#define XT(F) \
kernel void x_##F(ARG) { \
    device const uint *u = tab + (130 + tg.y)*10; \
    uint c = u[1], m = (l/4 & 4) + l/2%4, n = (l/4 & 2)*2 + l%2*2, \
         gb = p.rb*256/p.n_in, r0 = tg.x*16 + m, \
         e0 = u[2 + min(n, c - 1)], e1 = u[2 + min(n + 1, c - 1)]; \
    if (!c) return; \
    W w0 = w + ((ulong)u[0]*p.n_out + r0)*p.rb; \
    X x0 = XU(e0) + (m & 1)*NH_##F, x1 = XU(e1) + (m & 1)*NH_##F; \
    simdgroup_float8x8 a, b[4], s[2][4]; \
    for (uint i = 0; i < 8; i++) \
        s[i/4][i%4] = make_filled_simdgroup_matrix<float, 8>(0.f); \
    for (uint k = 0; k < p.n_in/32; k++) { \
        uint g = k/8, q = k%8, o = 256*g + xo_##F(4*q + m/2); \
        float4 xa = *(X4)(x0 + o), xb = *(X4)(x1 + o); \
        for (uint i = 0; i < 4; i++) TE(b[i]) = float2(xa[i], xb[i]); \
        for (uint j = 0; j < 2; j++) { \
            float4 va, vb; \
            float d = d_##F(w0 + 8*j*p.rb + g*gb, 4*q + n/2, va, vb); \
            va *= d, vb *= d; \
            for (uint i = 0; i < 4; i++) { \
                TE(a) = float2(va[i], vb[i]); \
                simdgroup_multiply_accumulate(s[j][i], a, b[i], s[j][i]); \
            } \
        } \
    } \
    for (uint j = 0; j < 2; j++) { \
        float2 v = TE(s[j][0]) + TE(s[j][1]) + (TE(s[j][2]) + TE(s[j][3])); \
        if (n < c) YU(e0)[r0 + 8*j] = v[0]; \
        if (n + 1 < c) YU(e1)[r0 + 8*j] = v[1]; \
    } \
}
// rows per threadgroup at 1 use, as XT in dr.rs; 4 at 2
#define XS(F, R) XI(F, R, 1) XI(F, 4, 2) XT(F)
XS(q8_0, 4) XS(q4_k, 2) XS(q5_k, 2) XS(q6_k, 2) XS(iq2_xxs, 8) XS(iq2_xs, 4)
XS(iq2_s, 8) XS(iq3_xxs, 8) XS(iq4_nl, 2) XS(nvfp4, 2)
