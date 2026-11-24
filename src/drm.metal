// d_<fmt>: the 8 weights of unit e of the group at b, dequantized once,
// and their common scale; their inputs are the float4s at xo_<fmt>(e) and
// NH_<fmt> after it. as in mv.metal, bytes and grid magnitudes come as
// subnormal halves (b 2^-24) and nvfp4 nibbles as e2m1 2^-14, with the
// powers of two in the scale: the products are those of the plain decode,
// bit for bit (q8_0 stays a convert: the half form ran slower). G: the
// format's table in threadgroup memory (TG_<fmt>)
#define DQ(F) static inline float d_##F(W b, uint e, thread float4 &va, \
    thread float4 &vb, threadgroup const ulong *G)
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
// the table of a format: ulongs, and their fill by the 32 lanes
#define TF(T, n, ty) for (uint i = l; i < n; i += 32) \
    ((threadgroup ty *)tgg)[i] = T[i];
#define TG_q8_0 1,
#define TG_q4_k 1,
#define TG_q5_k 1,
#define TG_q6_k 1,
#define TG_nvfp4 1,
#define TG_iq2_xxs 256, TF(iq2xxs_grid, 256, ulong)
#define TG_iq2_xs 512, TF(iq2xs_grid, 512, ulong)
#define TG_iq2_s 1, // 8 KB: a copy a threadgroup costs more than it saves
#define TG_iq3_xxs 128, TF(iq3xxs_grid, 256, uint)
#define TG_iq4_nl 128, for (uint i = l; i < 256; i += 32) \
    ((threadgroup half2 *)tgg)[i] = half2(k_iq4nl[i & 15], k_iq4nl[i >> 4]);
#define TGA(n, ...) threadgroup ulong tgg[n]; __VA_ARGS__ \
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define TGX(...) TGA(__VA_ARGS__)
#define W4(p) (*(device const uint *)(p))

// a signed byte b as the half of b + 128, less 128 2^-24
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
    W q = b + 16 + 32*(j/2) + 8*(e%4);
    va = d.x*0x1p24f*HU(W4(q) >> sh & 0x0f0f0f0f) - d.y;
    vb = d.x*0x1p24f*HU(W4(q + 4) >> sh & 0x0f0f0f0f) - d.y;
    return 1;
}

DQ(q5_k) {
    uint j = e/4, sh = 4*(j & 1), o = 8*(e%4);
    float2 d = D2*k4(b + 4, j);
    W h = b + 16 + o, q = b + 48 + 32*(j/2) + o;
    va = d.x*0x1p24f*HU((W4(q) >> sh & 0x0f0f0f0f) |
                        (W4(h) >> j & 0x01010101) << 4) - d.y;
    vb = d.x*0x1p24f*HU((W4(q + 4) >> sh & 0x0f0f0f0f) |
                        (W4(h + 4) >> j & 0x01010101) << 4) - d.y;
    return 1;
}

// the k-quant's 6-bit values less 32
DQ(q6_k) {
    uint n = e/16, g = (e/4)%4, o = 8*(e%4), sh = 4*(g/2);
    uint2 q = (h2(b + 64*n + 32*(g & 1) + o) >> sh & 0x0f0f0f0f) |
              (h2(b + 128 + 32*n + o) >> 2*g & 0x03030303) << 4;
    va = HU(q.x) - 0x1p-19f, vb = HU(q.y) - 0x1p-19f;
    return F16(b + 208)*0x1p24f*
        (float)((device const char *)b)[192 + 8*n + 2*g + o/16];
}

// both nibbles of a byte, as halves, from G
DQ(iq4_nl) {
    W c = b + (e/4)*18;
    uint4 q = uint4(*(Q)(c + 2 + 4*(e%4)));
    threadgroup const half2 *t = (threadgroup const half2 *)G;
    half2 v0 = t[q.x], v1 = t[q.y], v2 = t[q.z], v3 = t[q.w];
    va = float4(v0.x, v1.x, v2.x, v3.x), vb = float4(v0.y, v1.y, v2.y, v3.y);
    return F16(c);
}

DQ(nvfp4) {
    W a = b + (e/8)*36;
    uint c = a[e%8/2], q = W4(a + 4 + 4*(e%8));
    va = hd4(F4(q, 0), F4(q, 8)), vb = hd4(F4(q, 4), F4(q, 12));
    return c == 127 ? 0 :
        (float)as_type<half>((ushort)((c & 127) << 7))*0x1p22f;
}

#define GRID(d, g0, g1, sg) \
    va = H4(uint(g0), uint(sg)*0x01000400, 5, 4); \
    vb = H4(uint(g1), uint(sg)*0x01000400, 1, 0); \
    return d

DQ(iq2_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 2 + 8*ib);
    uint aux = a[2] | (uint(a[3]) << 16);
    ulong g = G[b[2 + 8*ib + l]];
    GRID(F16(b)*(0.5f + (aux >> 28))*0x1p22f, g, g >> 32,
         ksigns_iq2xs[(aux >> 7*l) & 127]);
}

DQ(iq2_xs) {
    uint q = ((H)(b + 2))[e];
    ulong g = G[q & 511];
    float ls = (b[66 + e/4] >> 4*((e%4)/2)) & 15;
    GRID(F16(b)*(0.5f + ls)*0x1p22f, g, g >> 32, ksigns_iq2xs[q >> 9]);
}

DQ(iq2_s) {
    ulong g = iq2s_grid[b[2 + e] | (((b[66 + e/4] >> 2*(e%4)) & 3) << 8)];
    float ls = (b[74 + e/4] >> 4*((e%4)/2)) & 15;
    GRID(F16(b)*(0.5f + ls)*0x1p22f, g, g >> 32, b[34 + e]);
}

DQ(iq3_xxs) {
    uint ib = e/4, l = e%4;
    H a = (H)(b + 66 + 4*ib);
    uint aux = a[0] | (uint(a[1]) << 16);
    threadgroup const uint *t = (threadgroup const uint *)G;
    GRID(F16(b)*(0.5f + (aux >> 28))*0x1p23f, t[b[2 + 2*e]],
         t[b[3 + 2*e]], ksigns_iq2xs[(aux >> 7*l) & 127]);
}

// units 2j, 2j + 1 with the loads of both at once, each as d_<fmt>:
// weights in v[0..1] and v[2..3], one scale
#define DQ2(F) static inline float d2_##F(W b, uint j, thread float4 *v, \
    threadgroup const ulong *G)
#define GR2(g0, g1, sg, v) \
    v[0] = H4(uint(g0), uint(sg)*0x01000400, 5, 4); \
    v[1] = H4(uint(g1), uint(sg)*0x01000400, 1, 0)
#define GRID2(g, sg, v) GR2(g, (g) >> 32, sg, v)

DQ2(iq2_s) {
    uint q = ((H)(b + 2))[j], sg = ((H)(b + 34))[j], c = 4*(j & 1),
         h = b[66 + j/2] >> c;
    GRID2(iq2s_grid[(q & 255) | (h & 3) << 8], sg & 255, v);
    GRID2(iq2s_grid[(q >> 8) | (h & 12) << 6], sg >> 8, (v + 2));
    return F16(b)*(0.5f + ((b[74 + j/2] >> c) & 15))*0x1p22f;
}

DQ2(iq2_xs) {
    uint q0 = ((H)(b + 2))[2*j], q1 = ((H)(b + 2))[2*j + 1];
    GRID2(G[q0 & 511], ksigns_iq2xs[q0 >> 9], v);
    GRID2(G[q1 & 511], ksigns_iq2xs[q1 >> 9], (v + 2));
    return F16(b)*(0.5f + ((b[66 + j/2] >> 4*(j & 1)) & 15))*0x1p22f;
}

DQ2(iq3_xxs) {
    H a = (H)(b + 66 + 4*(j/2));
    uint aux = a[0] | (uint(a[1]) << 16), l = 2*(j & 1),
         q0 = ((H)b)[1 + 2*j], q1 = ((H)b)[2 + 2*j];
    threadgroup const uint *t = (threadgroup const uint *)G;
    GR2(t[q0 & 255], t[q0 >> 8], ksigns_iq2xs[(aux >> 7*l) & 127], v);
    GR2(t[q1 & 255], t[q1 >> 8], ksigns_iq2xs[(aux >> 7*(l + 1)) & 127],
        (v + 2));
    return F16(b)*(0.5f + (aux >> 28))*0x1p23f;
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
    TGX(TG_##F) \
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
            float d = d_##F(w0 + i*p.rb + k*gb, l, a, b, tgg); \
            for (uint j = 0; j < C; j++) \
                s[i][j] += d*(dot(a, xa[j]) + dot(b, xb[j])); \
        } \
    } \
    for (uint i = 0; i < R*C; i++) { \
        float v = simd_sum(s[i/C][i%C]); \
        if (l == 0) YU(u[2 + i%C])[r0 + i/C] = v; \
    } \
}

// as XI, half the weight loads: lane l takes units 2j, 2j + 1 (j = l%16)
// of rows R/2 (l/16).. of every group. the shuffles give lane u unit u's
// sums, so simd_sum adds what XI's lanes held: the same bits
#define XI2(F, R, C) \
kernel void x_##F##_##C(ARG) { \
    device const uint *u = tab + (65*(C - 1) + tg.y)*10; \
    if (!u[1]) return; \
    TGX(TG_##F) \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*R, j = l%16; \
    W w0 = w + ((ulong)u[0]*p.n_out + r0 + l/16*R/2)*p.rb; \
    X4 xp[C]; \
    for (uint c = 0; c < C; c++) xp[c] = (X4)(XU(u[2 + c]) + 16*j); \
    float s[R/2][2][C] = {}; \
    for (uint k = 0; 32*k + 2*j < U; k++) { \
        float4 x[C][4]; \
        for (uint c = 0; c < C; c++) \
            for (uint i = 0; i < 4; i++) x[c][i] = xp[c][64*k + i]; \
        for (uint i = 0; i < R/2; i++) { \
            float4 v[4]; \
            float d = d2_##F(w0 + i*p.rb + k*gb, j, v, tgg); \
            for (uint c = 0; c < C; c++) { \
                s[i][0][c] += d*(dot(v[0], x[c][0]) + dot(v[1], x[c][1])); \
                s[i][1][c] += d*(dot(v[2], x[c][2]) + dot(v[3], x[c][3])); \
            } \
        } \
    } \
    for (uint i = 0; i < R/2*C; i++) { \
        float2 v = float2(s[i/C][0][i%C], s[i/C][1][i%C]), \
               a = simd_shuffle(v, l/2), b = simd_shuffle(v, 16 + l/2); \
        float y0 = simd_sum(a[l%2]), y1 = simd_sum(b[l%2]); \
        if (l == 0) YU(u[2 + i%C])[r0 + i/C] = y0, \
                    YU(u[2 + i%C])[r0 + R/2 + i/C] = y1; \
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
    TGX(TG_##F) \
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
            float d = d_##F(w0 + 8*j*p.rb + g*gb, 4*q + n/2, va, vb, tgg); \
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
// rows per threadgroup at 1 and 2 uses, as XT in dr.rs
#define XS(F, R) XI(F, R, 1) XI(F, 4, 2) XT(F)
XS(q8_0, 4) XS(q4_k, 2) XS(q5_k, 2) XS(q6_k, 2) XS(iq2_xxs, 8) XS(iq4_nl, 2)
XS(nvfp4, 2) XI2(iq2_xs, 8, 1) XI2(iq2_xs, 8, 2) XT(iq2_xs) XI2(iq2_s, 8, 1)
XI2(iq2_s, 8, 2) XT(iq2_s) XI2(iq3_xxs, 8, 1) XI(iq3_xxs, 4, 2) XT(iq3_xxs)
