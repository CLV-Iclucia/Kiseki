// ============================================================================
// scan-propagate.hlsl — Add block offsets to local elements
//
// Must match ITEMS_PER_THREAD from scan-local.hlsl.
// Each thread handles ITEMS_PER_THREAD elements.
// ============================================================================

#include "rpk-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<scalar_t> data         : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<scalar_t>   blockOffsets : register(t0);

struct PushParams {
    uint numElements;
};
[[vk::push_constant]] PushParams pc;

#define ITEMS_PER_THREAD 16
#define ELEMENTS_PER_GROUP (WORKGROUP_SIZE * ITEMS_PER_THREAD)

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    // First group has no offset to add
    if (groupId.x == 0) return;

    scalar_t offset = blockOffsets[groupId.x];

    uint blockBase = groupId.x * ELEMENTS_PER_GROUP;
    uint baseIdx = blockBase + gid * ITEMS_PER_THREAD;

    for (uint i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint idx = baseIdx + i;
        if (idx < pc.numElements) {
            data[idx] = rpk_op(offset, data[idx]);
        }
    }
}
