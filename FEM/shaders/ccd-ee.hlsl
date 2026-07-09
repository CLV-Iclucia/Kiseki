// ============================================================================
// ccd-ee.hlsl — module D, edge-edge additive CCD.
// One thread per EE candidate {a0, a1, b0, b1}; same as ccd-vt but EE mode.
// ============================================================================
#include <fem/shared/ipc-ccd.h>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> toiOut   : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   x        : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   pdir     : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint>     cand     : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<double>   paramBuf : register(t3); // [toiCap, s]

struct PC { uint num; uint streamOffset; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.num) return;
    uint e = pc.streamOffset + i;

    int a0 = (int)cand[i * 4 + 0];
    int a1 = (int)cand[i * 4 + 1];
    int b0 = (int)cand[i * 4 + 2];
    int b1 = (int)cand[i * 4 + 3];

    double3 x0 = load_double3(x, a0);
    double3 x1 = load_double3(x, a1);
    double3 x2 = load_double3(x, b0);
    double3 x3 = load_double3(x, b1);

    double3 u0 = load_double3(pdir, a0);
    double3 u1 = load_double3(pdir, a1);
    double3 u2 = load_double3(pdir, b0);
    double3 u3 = load_double3(pdir, b1);

    toiOut[e] = shACCD(SH_CCD_EE, x0, x1, x2, x3, u0, u1, u2, u3,
                       paramBuf[0], paramBuf[1]);
}
