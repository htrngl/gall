// several inputs against one read of the weights, on mv.metal's units
// (appended to it by dr.rs): y[t][r] = W[r] x[t] for t < nt <= 8. x rows
// are n_in apart, y rows ys apart. lanes and rows as in mv.metal.
struct PM { uint n_in, n_out, rb, nt, ys; };

// a unit is dequantized once for all inputs (inside mv.metal's u_<fmt>
// the compiler repeats it per input: 5x slower), and the input count T is
// a constant (a run-time count: 2x slower). kernels mm_<fmt>_<T>.
// q8_0, 4 rows
#define MM8(T) \
kernel void mm_q8_0_##T(constant PM &p, W w [[buffer(1)]], X x [[buffer(2)]], \
        device float *y [[buffer(3)]], \
        uint3 tg [[threadgroup_position_in_grid]], \
        ushort l [[thread_index_in_threadgroup]]) { \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*4; \
    float4 s[T] = {0}; \
    for (uint k = 0; 32*k + l < U; k++) { \
        float4 a[4], b[4]; \
        for (uint i = 0; i < 4; i++) { \
            W c = w + (ulong)(r0 + i)*p.rb + k*gb + (l/4)*34; \
            device const packed_char4 *q = \
                (device const packed_char4 *)(c + 2 + 8*(l%4)); \
            a[i] = F16(c)*float4(char4(q[0])); \
            b[i] = F16(c)*float4(char4(q[1])); \
        } \
        for (uint t = 0; t < T; t++) { \
            device const packed_float4 *xp = (device const packed_float4 *) \
                (x + t*p.n_in + 256*k + 8*l); \
            float4 xa = xp[0], xb = xp[1]; \
            s[t] += float4(dot(a[0], xa) + dot(b[0], xb), \
                           dot(a[1], xa) + dot(b[1], xb), \
                           dot(a[2], xa) + dot(b[2], xb), \
                           dot(a[3], xa) + dot(b[3], xb)); \
        } \
    } \
    for (uint t = 0; t < T; t++) \
        for (uint i = 0; i < 4; i++) { \
            float v = simd_sum(s[t][i]); \
            if (l == 0) y[t*p.ys + r0 + i] = v; \
        } \
}

// q5_k, 2 rows
#define MM5(T) \
kernel void mm_q5_k_##T(constant PM &p, W w [[buffer(1)]], X x [[buffer(2)]], \
        device float *y [[buffer(3)]], \
        uint3 tg [[threadgroup_position_in_grid]], \
        ushort l [[thread_index_in_threadgroup]]) { \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*2; \
    uint j = l/4, sh = 4*(j & 1), o = 8*(l%4); \
    float2 s[T] = {0}; \
    for (uint k = 0; 32*k + l < U; k++) { \
        float4 a[2], b[2]; \
        for (uint i = 0; i < 2; i++) { \
            W c = w + (ulong)(r0 + i)*p.rb + k*gb; \
            float2 d = float2(*(device const packed_half2 *)c)*k4(c + 4, j); \
            Q h = (Q)(c + 16 + o), q = (Q)(c + 48 + 32*(j/2) + o); \
            a[i] = d.x*float4(((uint4(q[0]) >> sh) & 15) + \
                              (((uint4(h[0]) >> j) & 1) << 4)) - d.y; \
            b[i] = d.x*float4(((uint4(q[1]) >> sh) & 15) + \
                              (((uint4(h[1]) >> j) & 1) << 4)) - d.y; \
        } \
        for (uint t = 0; t < T; t++) { \
            device const packed_float4 *xp = (device const packed_float4 *) \
                (x + t*p.n_in + 256*k + 8*l); \
            float4 xa = xp[0], xb = xp[1]; \
            s[t] += float2(dot(a[0], xa) + dot(b[0], xb), \
                           dot(a[1], xa) + dot(b[1], xb)); \
        } \
    } \
    for (uint t = 0; t < T; t++) \
        for (uint i = 0; i < 2; i++) { \
            float v = simd_sum(s[t][i]); \
            if (l == 0) y[t*p.ys + r0 + i] = v; \
        } \
}

MM8(8)
MM5(1) MM5(2) MM5(3) MM5(4) MM5(5) MM5(6) MM5(7) MM5(8)

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

// C activation rows against one read of an expert slab (or of a dense
// matrix: slab 0). entry tg.y of class C in tab (dr.metal uni: 65 entries
// of 9 per class) is the slab and C uses, row t << 3 | k:
// y[t][k][r] = W[slab][r] x[t][k]. x rows are xs apart and x[t][k] is kx
// floats after x[t][0]; y likewise with ys and n_out. R weight rows per
// threadgroup. the use count is a constant: a slab is read once and costs
// one dot per use
struct PX { uint n_in, n_out, rb, xs, ys, kx; };
#define XK(N, F, R, C, ...) \
kernel void N##_##F##_##C(constant PX &p, W w [[buffer(1)]], \
        X x [[buffer(2)]], device float *y [[buffer(3)]], \
        device const uint *tab [[buffer(4)]], \
        uint3 tg [[threadgroup_position_in_grid]], \
        ushort l [[thread_index_in_threadgroup]]) { \
    device const uint *u = tab + (65*(C - 1) + tg.y)*9; \
    if (u[0] == ~0u) return; \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*R; \
    W w0 = w + ((ulong)u[0]*p.n_out + r0)*p.rb; \
    device const packed_float4 *xp[C]; \
    for (uint j = 0; j < C; j++) \
        xp[j] = (device const packed_float4 *)(x + (u[1 + j] >> 3)*p.xs + \
            (u[1 + j] & 7)*p.kx + xo_##F(l)); \
    float s[R*C] = {0}; \
    for (uint k = 0; 32*k + l < U; k++) { __VA_ARGS__ } \
    for (uint i = 0; i < R*C; i++) { \
        float v = simd_sum(s[i]); \
        uint e = u[1 + i%C]; \
        if (l == 0) y[(e >> 3)*p.ys + (e & 7)*p.n_out + r0 + i/C] = v; \
    } \
}
// XI: the inputs are held and the rows dequantized in turn; XW: the rows
// are held and the inputs loaded in turn
#define XI(F, R, C) XK(x, F, R, C, \
    float4 xa[C], xb[C]; \
    for (uint j = 0; j < C; j++) \
        xa[j] = xp[j][64*k], xb[j] = xp[j][64*k + NH_##F/4]; \
    for (uint i = 0; i < R; i++) { \
        float4 a, b; \
        float d = d_##F(w0 + i*p.rb + k*gb, l, a, b); \
        for (uint j = 0; j < C; j++) \
            s[i*C + j] += d*(dot(a, xa[j]) + dot(b, xb[j])); \
    })
#define XW(F, R, C) XK(x, F, R, C, \
    float4 a[R], b[R]; \
    float d[R]; \
    for (uint i = 0; i < R; i++) \
        d[i] = d_##F(w0 + i*p.rb + k*gb, l, a[i], b[i]); \
    for (uint j = 0; j < C; j++) { \
        float4 xa = xp[j][64*k], xb = xp[j][64*k + NH_##F/4]; \
        for (uint i = 0; i < R; i++) \
            s[i*C + j] += d[i]*(dot(a[i], xa) + dot(b[i], xb)); \
    })
// rows per threadgroup for 1, 2 and 3 to 6 uses, as in dr.rs; 2 above.
// measured in a pass of 4 rows, shares in ms at 8 / 4 / 2 rows: iq3_xxs
// 1 use 10.3 / 10.5 / 11.9, 2 uses 7.0 / 5.6 / 6.3, 3 uses 7.1 / 5.2 / 4.5;
// nvfp4 3 uses 7.5 / 4.8 / 3.2; q8_0 4 uses 20.4 / 17.9 / 20.3. 8 uses of
// q8_0 in a pass of 8: XI 55 at 4 rows, 61 at 2; XW 95 and 47
#define XS(F, A, B, D) XI(F, A, 1) XI(F, B, 2) XI(F, D, 3) XI(F, D, 4) \
    XI(F, D, 5) XI(F, D, 6) XW(F, 2, 7) XW(F, 2, 8)
XS(q8_0, 4, 4, 4) XS(q4_k, 2, 4, 2) XS(q5_k, 2, 4, 2) XS(q6_k, 2, 4, 2)
XS(iq2_xxs, 8, 4, 2) XS(iq2_xs, 4, 4, 2) XS(iq2_s, 8, 4, 2)
XS(iq3_xxs, 8, 4, 2) XS(iq4_nl, 4, 4, 2) XS(nvfp4, 8, 4, 2)
