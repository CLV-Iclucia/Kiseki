// ============================================================================
// reduce.hlsl — Per-workgroup parallel reduction (level 0)
// Outputs one partial result per workgroup into outputBuf[groupId.x].
// ============================================================================

#include "rpk-common.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<scalar_t>   inputBuf  : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<scalar_t> outputBuf : register(u0);

struct PushParams {
    uint numElements;
};
[[vk::push_constant]] PushParams pc;

groupshared scalar_t sdata[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    // Each thread loads one element (or identity if out of bounds)
    scalar_t val = OP_IDENTITY;
    if (dtid.x < pc.numElements) {
        val = inputBuf[dtid.x];
    }

    sdata[gid] = val;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction
    [unroll]
    for (uint s = WORKGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (gid < s) {
            sdata[gid] = rpk_op(sdata[gid], sdata[gid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) {
        outputBuf[groupId.x] = sdata[0];
    }
}
