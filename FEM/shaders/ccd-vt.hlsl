// ============================================================================
// ccd-vt.hlsl — module D, vertex-triangle additive CCD.
// One thread per VT candidate {v, t0, t1, t2}. Run shared ACCD with the full
// Newton step direction; write the per-candidate time-of-impact (or the cap if
// the pair does not limit the step) into toiOut[streamOffset + i].
// A global Min reduction over toiOut yields the conservative step bound.
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

    int v  = (int)cand[i * 4 + 0];
    int t0 = (int)cand[i * 4 + 1];
    int t1 = (int)cand[i * 4 + 2];
    int t2 = (int)cand[i * 4 + 3];

    double3 x0 = load_double3(x, v);
    double3 x1 = load_double3(x, t0);
    double3 x2 = load_double3(x, t1);
    double3 x3 = load_double3(x, t2);

    double3 u0 = load_double3(pdir, v);
    double3 u1 = load_double3(pdir, t0);
    double3 u2 = load_double3(pdir, t1);
    double3 u3 = load_double3(pdir, t2);

    toiOut[e] = shACCD(SH_CCD_VT, x0, x1, x2, x3, u0, u1, u2, u3,
                       paramBuf[0], paramBuf[1]);
}
