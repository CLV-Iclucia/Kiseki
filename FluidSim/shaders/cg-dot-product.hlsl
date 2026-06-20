// ============================================================================
// cg-dot-product.hlsl — CG: dot product with groupshared reduction (float32)
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<float>   vecA      : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   vecB      : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> reduceBuf : register(u0);

struct PushParams {
    uint numCells;
};
[[vk::push_constant]] PushParams pc;

groupshared float sharedData[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex, uint3 grpId : SV_GroupID) {
    uint idx = tid.x;
    sharedData[gid] = (idx < pc.numCells) ? vecA[idx] * vecB[idx] : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) sharedData[gid] += sharedData[gid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) reduceBuf[grpId.x] = sharedData[0];
}
