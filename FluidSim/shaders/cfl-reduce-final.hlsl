// ============================================================================
// cfl-reduce-final.hlsl — Final reduction: max of partial workgroup results
// Reads partialMax[0..numGroups-1], writes max speed to scalarOut[0].
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<float>   partialMax : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> scalarOut  : register(u0);

struct PushParams {
    uint numGroups;
};
[[vk::push_constant]] PushParams pc;

groupshared float shared_max[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    shared_max[gid] = (gid < pc.numGroups) ? partialMax[gid] : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) shared_max[gid] = max(shared_max[gid], shared_max[gid + s]);
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) scalarOut[0] = shared_max[0];
}
