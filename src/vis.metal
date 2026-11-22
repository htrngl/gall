// the vision encoder's small kernels (vis.rs); its weights go through
// mm_f16 of pfm.metal. a: f32 activations, VS floats per position;
// s: small f32 tensors; one thread per position unless said.
#include <metal_stdlib>
using namespace metal;

struct VP { uint n, x, y, w, b; };
typedef device const packed_float4 *F4;

// y = layer norm of x (1024) times s[w..] plus s[b..]
kernel void vln(constant VP &p, device float *a [[buffer(1)]],
        device const float *s [[buffer(2)]],
        uint t [[thread_position_in_grid]]) {
    if (t >= p.n) return;
    device float *x = a + (ulong)t*VS + p.x, *y = a + (ulong)t*VS + p.y;
    float m = 0, v = 0;
    for (uint i = 0; i < 1024; i++) m += x[i];
    m /= 1024;
    for (uint i = 0; i < 1024; i++) v += (x[i] - m)*(x[i] - m);
    v = rsqrt(v/1024 + 1e-5f);
    for (uint i = 0; i < 1024; i++)
        y[i] = (x[i] - m)*v*s[p.w + i] + s[p.b + i];
}

// head t.y of q, k, v at x: plus the biases at s[b..]; q and k turn by
// the position's 32 (sin, cos) pairs in r
kernel void vqkv(constant VP &p, device float *a [[buffer(1)]],
        device const float *s [[buffer(2)]],
        device const float2 *r [[buffer(3)]],
        uint2 t [[thread_position_in_grid]]) {
    if (t.x >= p.n) return;
    for (uint j = 0; j < 3; j++) {
        device float *x = a + (ulong)t.x*VS + p.x + 1024*j + 64*t.y;
        device const float *b = s + p.b + 1024*j + 64*t.y;
        for (uint i = 0; i < 64; i += 2) {
            float2 v = float2(x[i] + b[i], x[i + 1] + b[i + 1]),
                   c = j < 2 ? r[t.x*32 + i/2] : float2(0, 1);
            x[i] = v.x*c.y - v.y*c.x;
            x[i + 1] = v.x*c.x + v.y*c.y;
        }
    }
}

// head tg.y of 32 queries over the n keys: y = softmax(q k / 8) v, with a
// running maximum. q at x, k and v after it. the threadgroup loads 32
// keys at a time, a key per thread, and every query reads them there
kernel void vatt(constant VP &p, device float *a [[buffer(1)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort t [[thread_index_in_threadgroup]]) {
    threadgroup float4 sk[512], sv[512];
    uint qi = tg.x*32 + t, h = p.x + 64*tg.y;
    F4 q = (F4)(a + (ulong)min(qi, p.n - 1)*VS + h);
    float4 qv[16], acc[16];
    for (uint i = 0; i < 16; i++) qv[i] = float4(q[i])*0.125f, acc[i] = 0;
    float m = -1e30f, sum = 0;
    for (uint j = 0; j < p.n; j += 32) {
        F4 k = (F4)(a + (ulong)min(j + t, p.n - 1)*VS + h + 1024);
        for (uint i = 0; i < 16; i++)
            sk[16*t + i] = k[i], sv[16*t + i] = k[256 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < min(32u, p.n - j); c++) {
            float4 d4 = 0;
            for (uint i = 0; i < 16; i++) d4 += qv[i]*sk[16*c + i];
            float d = d4.x + d4.y + d4.z + d4.w;
            if (d > m) {
                float f = exp(m - d);
                sum *= f, m = d;
                for (uint i = 0; i < 16; i++) acc[i] *= f;
            }
            float w = exp(d - m);
            sum += w;
            for (uint i = 0; i < 16; i++) acc[i] += w*sv[16*c + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (qi >= p.n) return;
    device packed_float4 *y =
        (device packed_float4 *)(a + (ulong)qi*VS + p.y + 64*tg.y);
    for (uint i = 0; i < 16; i++) y[i] = acc[i]/sum;
}

// x += (y + s[b..]) * s[w..]
kernel void vres(constant VP &p, device float *a [[buffer(1)]],
        device const float *s [[buffer(2)]],
        uint t [[thread_position_in_grid]]) {
    if (t >= p.n) return;
    device float *x = a + (ulong)t*VS + p.x, *y = a + (ulong)t*VS + p.y;
    for (uint i = 0; i < 1024; i++) x[i] += (y[i] + s[p.b + i])*s[p.w + i];
}

// 64 of x at t.y: gelu(x + s[b..]) (erf form; erf to 1.5e-7, a&s 7.1.26);
// w = 0: no bias
kernel void vgelu(constant VP &p, device float *a [[buffer(1)]],
        device const float *s [[buffer(2)]],
        uint2 t [[thread_position_in_grid]]) {
    if (t.x >= p.n) return;
    device float *x = a + (ulong)t.x*VS + p.x + 64*t.y;
    for (uint i = 0; i < 64; i++) {
        float v = x[i] + (p.w ? s[p.b + 64*t.y + i] : 0),
              z = abs(v)*0.70710678f, u = 1/(1 + 0.3275911f*z),
              e = 1 - ((((1.061405429f*u - 1.453152027f)*u + 1.421413741f)*u
                  - 0.284496736f)*u + 0.254829592f)*u*exp(-z*z);
        x[i] = 0.5f*v*(1 + copysign(e, v));
    }
}
