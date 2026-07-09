// ============================================================================
// reduce-final.hlsl — Final reduction: single workgroup reduces all inputs
// to a single scalar in outputBuf[0].
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
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    // Sequential accumulation if numElements > WORKGROUP_SIZE
    scalar_t val = OP_IDENTITY;
    for (uint i = gid; i < pc.numElements; i += WORKGROUP_SIZE) {
        val = rpk_op(val, inputBuf[i]);
    }

    sdata[gid] = val;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = WORKGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (gid < s) {
            sdata[gid] = rpk_op(sdata[gid], sdata[gid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gid == 0) {
        outputBuf[0] = sdata[0];
    }
}
