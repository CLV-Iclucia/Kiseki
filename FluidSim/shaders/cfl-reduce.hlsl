// ============================================================================
// cfl-reduce.hlsl — Parallel reduction: compute max particle speed (float32)
// Two-pass: this shader does per-workgroup max, writing partial results.
// Final reduction is done by cfl-reduce-final.hlsl.
// ============================================================================

#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<float> velocities : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partialMax : register(u0);

struct PushParams {
    uint numParticles;
};
[[vk::push_constant]] PushParams pc;

groupshared float shared_max[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    float localMax = 0.0f;
    uint idx = dtid.x;
    if (idx < pc.numParticles) {
        uint b = idx * 3;
        float3 v = float3(velocities[b], velocities[b+1], velocities[b+2]);
        localMax = length(v);
    }

    shared_max[gid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction within workgroup (max)
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) {
            shared_max[gid] = max(shared_max[gid], shared_max[gid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) {
        partialMax[groupId.x] = shared_max[0];
    }
}
