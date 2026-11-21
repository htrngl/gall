// weight mat-mat for prefill, appended to mv.metal. Y = W X by tiles of
// 64 rows x 32 columns, a threadgroup of 4 simdgroups each (2 row halves
// x 2 column halves). per 32 inputs a thread dequantizes 16 weights
// (u_<fmt> against unit vectors) and loads 8 inputs into threadgroup
// tiles of 8x8, then the simdgroups multiply them. tile tg.y is
// (expert, first column, columns <= 32, a multiple of 8) in tl. column c
// reads x at xo + c*xs, or at xo + (col[c] & 0xffff)*xs with ls, and
// writes y at yo + c*ys. rows past n_out are computed from what follows
// W and written too: y has room for a multiple of 64.

struct PM { uint n_in, n_out, rb, ls, xo, xs, yo, ys; };

// tile precision. float tiles: 0.8x the speed; logits after 2200 tokens
// against stepping kl 7e-4, max abs 0.5 (half: 8e-3, 1.6)
#ifndef HT
#define HT half
#define HM simdgroup_half8x8
#endif
#define TGB threadgroup_barrier(mem_flags::mem_threadgroup)

// multiply and store for n column tiles: constant loop bounds, or the
// simdgroup matrices leave the registers (7x slower)
#define MUL(n) \
    for (uint ik = 0; ik < 4; ik++, la += 512, lb += 256) { \
        for (uint i = 0; i < 4; i++) simdgroup_load(ma[i], la + 64*i, 8); \
        for (uint i = 0; i < n; i++) simdgroup_load(mb[i], lb + 64*i, 8); \
        for (uint i = 0; i < 4*n; i++) \
            simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); \
    }
#define PUT(n) \
    for (uint i = 0; i < 4*n; i++) \
        simdgroup_store(mc[i], y + 8*(i%4) + (ulong)8*p.ys*(i/4), p.ys);

#define MM(F) \
kernel void mm_##F(constant PM &p, W w [[buffer(1)]], X xb [[buffer(2)]], \
        device float *yb [[buffer(3)]], device const uint *col [[buffer(4)]], \
        device const uint *tl [[buffer(5)]], \
        uint3 tg [[threadgroup_position_in_grid]], \
        ushort t [[thread_index_in_threadgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup HT sa[64*32], sb[32*32]; \
    uint ex = tl[3*tg.y], c0 = tl[3*tg.y + 1], nc = tl[3*tg.y + 2], \
         gb = p.rb*256/p.n_in, c = c0 + min(t/4u, nc - 1); \
    X x = xb + p.xo + (ulong)(p.ls ? col[c] & 0xffff : c)*p.xs + 8*(t%4); \
    W wr = w + ((ulong)ex*p.n_out + tg.x*64 + t/2)*p.rb; \
    threadgroup HT *pa = sa + 64*(16*(t%2) + t/16) + t/2%8, \
                   *pb = sb + 64*(4*(t%4) + t/32) + 8*(t/4%8); \
    int na = clamp((int)nc/8 - sg/2*2, 0, 2); \
    HM ma[4], mb[2]; \
    simdgroup_float8x8 mc[8]; \
    for (uint i = 0; i < 8; i++) \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); \
    for (uint k = 0; k < p.n_in; k += 32) { \
        for (uint u = 0; u < 2; u++) { \
            float4 z = 0, lo, hi; \
            for (uint i = 0; i < 4; i++) { \
                float4 v = 0; \
                v[i] = 1; \
                lo[i] = u_##F(wr + k/256*gb, k%256/8 + 2*(t%2) + u, v, z); \
                hi[i] = u_##F(wr + k/256*gb, k%256/8 + 2*(t%2) + u, z, v); \
            } \
            for (uint i = 0; i < 4; i++) \
                pa[512*u + 8*i] = lo[i], pa[512*u + 32 + 8*i] = hi[i]; \
        } \
        for (uint i = 0; i < 8; i++) pb[i] = x[k + i]; \
        TGB; \
        threadgroup const HT *la = sa + 256*(sg%2), *lb = sb + 128*(sg/2); \
        if (na == 2) MUL(2) else if (na == 1) MUL(1) \
        TGB; \
    } \
    device float *y = yb + p.yo + (ulong)(c0 + 16*(sg/2))*p.ys + tg.x*64 + \
                      32*(sg%2); \
    if (na == 2) PUT(2) else if (na == 1) PUT(1) \
}

// the small matrices of model.rs, ieee halfs
UNIT(f16) {
    device const packed_half4 *a = (device const packed_half4 *)(b + 16*e);
    return DOT8(a[0], a[1]);
}

MM(f16)
MM(q8_0)
MM(q4_k)
MM(q5_k)
MM(q6_k)
MM(iq2_xxs)
MM(iq2_xs)
MM(iq2_s)
MM(iq3_xxs)
MM(iq4_nl)
MM(nvfp4)
