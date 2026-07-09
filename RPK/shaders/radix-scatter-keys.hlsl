// ============================================================================
// radix-scatter-keys.hlsl — Radix sort: scatter keys only (no values)
// ============================================================================

#define WORKGROUP_SIZE 256
#define NUM_BUCKETS 16

[[vk::binding(0, 0)]] StructuredBuffer<uint>   keysIn     : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<uint>   prefixSums : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> keysOut    : register(u0);

struct PushParams {
    uint numElements;
    uint bitOffset;
    uint numGroups;
};
[[vk::push_constant]] PushParams pc;

groupshared uint sDigits[WORKGROUP_SIZE];
groupshared uint sLocalOffsets[NUM_BUCKETS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    uint key = 0;
    uint digit = 0;
    bool active = (dtid.x < pc.numElements);

    if (active) {
        key = keysIn[dtid.x];
        digit = (key >> pc.bitOffset) & 0xF;
    }
    sDigits[gid] = digit;

    if (gid < NUM_BUCKETS) {
        sLocalOffsets[gid] = prefixSums[gid * pc.numGroups + groupId.x];
    }
    GroupMemoryBarrierWithGroupSync();

    if (!active) return;

    // Compute local rank
    uint localRank = 0;
    for (uint i = 0; i < gid; ++i) {
        if (sDigits[i] == digit) {
            localRank++;
        }
    }

    uint outIdx = sLocalOffsets[digit] + localRank;
    keysOut[outIdx] = key;
}
