// ============================================================================
// lbvh-bound-block.hlsl — per-workgroup AABB reduction (scene-bound, pass 1).
// One thread per primitive AABB; block-reduces min(lo)/max(hi) per axis into
// partial[group*6 + {0..2}=lo, {3..5}=hi]. min/max are order-independent and
// exact, so the result matches CPU's serial sceneBound union bit-for-bit.
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> partial : register(u0); // [numGroups*6]
[[vk::binding(1, 0)]] StructuredBuffer<double>   lo      : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   hi      : register(t1);

struct PC { uint n; };   // number of primitives
[[vk::push_constant]] PC pc;

groupshared double gMin[256 * 3];
groupshared double gMax[256 * 3];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID,
          uint3 gtid : SV_GroupThreadID,
          uint3 gid  : SV_GroupID) {
    uint t = gtid.x;
    uint i = dtid.x;
    double3 local_min = (i < pc.n) ? load_double3(lo, i) : (double3)1e300;
    double3 local_max = (i < pc.n) ? load_double3(hi, i) : (double3)-1e300;
    [unroll] for (uint d = 0; d < 3u; ++d) {
        gMin[t * 3 + d] = local_min[d];
        gMax[t * 3 + d] = local_max[d];
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
        store_double3(partial, gid.x * 2u, double3(gMin[0], gMin[1], gMin[2]));
        store_double3(partial, gid.x * 2u + 1u, double3(gMax[0], gMax[1], gMax[2]));
    }
}
