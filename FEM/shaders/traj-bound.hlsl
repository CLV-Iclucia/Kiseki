// ============================================================================
// traj-bound.hlsl — per-primitive trajectory AABB.
// For each primitive (vertsPerPrim vertices via `conn`), the box is the union
// over its vertices of {x_v, x_v + alpha*p_v}. Mirrors the CPU trajectory bbox
// (collision-detector computeTrajectoryBBox / trajectory accessor): NO dHat
// dilation here — that is applied on the query side only.
//   outLo/outHi : double[numPrims*3]
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> outLo    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<double> outHi    : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<double>   x        : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<double>   p        : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<uint>     conn     : register(t2);
[[vk::binding(5, 0)]] StructuredBuffer<double>   alphaBuf : register(t3); // [1]

struct PC { uint numPrims; uint vertsPerPrim; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.numPrims) return;
    double alpha = alphaBuf[0];

    double3 lo = double3(1e300, 1e300, 1e300);
    double3 hi = double3(-1e300, -1e300, -1e300);

    for (uint j = 0; j < pc.vertsPerPrim; ++j) {
        uint v = conn[i * pc.vertsPerPrim + j];
        double3 start = load_double3(x, v);
        double3 end = start + alpha * load_double3(p, v);
        [unroll] for (uint d = 0; d < 3u; ++d) {
            lo[d] = min(lo[d], min(start[d], end[d]));
            hi[d] = max(hi[d], max(start[d], end[d]));
        }
    }
    store_double3(outLo, i, lo);
    store_double3(outHi, i, hi);
}
