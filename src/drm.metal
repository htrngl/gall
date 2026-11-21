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
MM5(1) MM5(2) MM5(3) MM5(4) MM5(7) MM5(8)

// T rows against one read of an expert slab (or of a dense matrix: slab
// 0), through u_<fmt>. slot tg.y of tab is (slab, k of row 0..T-1):
// y[t][k][r] = W[slab][r] x[t][k]; k = 255: row t does not use the slab.
// x rows are xs apart and x[t][k] is kx floats after x[t][0]; y likewise
// with ys and n_out. 2 weight rows per threadgroup
struct PX { uint n_in, n_out, rb, xs, ys, kx; };
#define XG(F, T) \
kernel void x_##F##_##T(constant PX &p, W w [[buffer(1)]], X x [[buffer(2)]], \
        device float *y [[buffer(3)]], device const uint *tab [[buffer(4)]], \
        uint3 tg [[threadgroup_position_in_grid]], \
        ushort l [[thread_index_in_threadgroup]]) { \
    device const uint *u = tab + tg.y*(T + 1); \
    if (u[0] == ~0u) return; \
    uint U = p.n_in/8, gb = p.rb*256/p.n_in, r0 = tg.x*2; \
    W w0 = w + ((ulong)u[0]*p.n_out + r0)*p.rb; \
    float s[2*T] = {0}; \
    for (uint k = 0; 32*k + l < U; k++) { \
        float4 xa[T], xb[T]; \
        for (uint t = 0; t < T; t++) { \
            device const packed_float4 *xp = (device const packed_float4 *) \
                (x + t*p.xs + (u[1 + t] & 7)*p.kx + 256*k + 8*l); \
            xa[t] = xp[0], xb[t] = xp[1]; \
        } \
        for (uint i = 0; i < 2; i++) \
            for (uint t = 0; t < T; t++) \
                s[i*T + t] += u_##F(w0 + i*p.rb + k*gb, l, xa[t], xb[t]); \
    } \
    for (uint i = 0; i < 2*T; i++) { \
        float v = simd_sum(s[i]); \
        uint t = i%T; \
        if (l == 0 && u[1 + t] != 255) \
            y[t*p.ys + u[1 + t]*p.n_out + r0 + i/T] = v; \
    } \
}
#define XGS(T) XG(q8_0, T) XG(q4_k, T) XG(q5_k, T) XG(q6_k, T) \
    XG(iq2_xxs, T) XG(iq2_xs, T) XG(iq2_s, T) XG(iq3_xxs, T) \
    XG(iq4_nl, T) XG(nvfp4, T)
XGS(2) XGS(4) XGS(8)
