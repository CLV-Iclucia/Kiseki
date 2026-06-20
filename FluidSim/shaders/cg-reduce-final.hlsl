// ============================================================================
// cg-reduce-final.hlsl — CG: Final reduction of partial sums to scalar (float32)
// Reads reduceBuf[0..numGroups-1], writes scalar to scalarOut[0].
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<float>   reduceBuf : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> scalarOut : register(u0);

struct PushParams {
    uint numGroups;
};
[[vk::push_constant]] PushParams pc;

groupshared float shared_data[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    shared_data[gid] = (gid < pc.numGroups) ? reduceBuf[gid] : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) shared_data[gid] += shared_data[gid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) scalarOut[0] = shared_data[0];
}
