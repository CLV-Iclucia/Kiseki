// ============================================================================
// lbvh-bound-final.hlsl — final scene-bound reduction (pass 2, single workgroup).
// Reduces the per-group partials [numGroups*6] into sceneBound[6] = {lo3, hi3}.
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> sceneBound : register(u0); // [6]
[[vk::binding(1, 0)]] StructuredBuffer<double>   partial    : register(t0); // [numGroups*6]

struct PC { uint numGroups; };
[[vk::push_constant]] PC pc;

groupshared double gMin[256 * 3];
groupshared double gMax[256 * 3];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint t = gtid.x;
    double3 lmin = (double3)1e300;
    double3 lmax = (double3)-1e300;
    for (uint g = t; g < pc.numGroups; g += 256u) {
        lmin = min(lmin, load_double3(partial, g * 2u));
        lmax = max(lmax, load_double3(partial, g * 2u + 1u));
    }
    [unroll] for (uint d = 0; d < 3u; ++d) {
        gMin[t * 3 + d] = lmin[d];
        gMax[t * 3 + d] = lmax[d];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) {
            [unroll] for (uint d = 0; d < 3u; ++d) {
                gMin[t * 3 + d] = min(gMin[t * 3 + d], gMin[(t + s) * 3 + d]);
                gMax[t * 3 + d] = max(gMax[t * 3 + d], gMax[(t + s) * 3 + d]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) {
        store_double3(sceneBound, 0u, double3(gMin[0], gMin[1], gMin[2]));
        store_double3(sceneBound, 1u, double3(gMax[0], gMax[1], gMax[2]));
    }
}
