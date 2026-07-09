// ============================================================================
// radix-scatter.hlsl — Radix sort: scatter key-value pairs to sorted positions
//
// Each thread computes its local rank within the workgroup for its digit,
// then uses globalOffset (from prefix sum) + localRank to write to output.
// ============================================================================

#define WORKGROUP_SIZE 256
#define NUM_BUCKETS 16

[[vk::binding(0, 0)]] StructuredBuffer<uint>   keysIn     : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<uint>   valuesIn   : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<uint>   prefixSums : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> keysOut    : register(u0);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> valuesOut  : register(u1);

struct PushParams {
    uint numElements;
    uint bitOffset;
    uint numGroups;
};
[[vk::push_constant]] PushParams pc;

// Shared memory: digits for all threads in this workgroup
groupshared uint sDigits[WORKGROUP_SIZE];
groupshared uint sLocalOffsets[NUM_BUCKETS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    // Load key and extract digit
    uint key = 0;
    uint value = 0;
    uint digit = 0;
    bool active = (dtid.x < pc.numElements);

    if (active) {
        key = keysIn[dtid.x];
        value = valuesIn[dtid.x];
        digit = (key >> pc.bitOffset) & 0xF;
    }
    sDigits[gid] = digit;

    // Load global prefix for this workgroup's buckets
    if (gid < NUM_BUCKETS) {
        sLocalOffsets[gid] = prefixSums[gid * pc.numGroups + groupId.x];
    }
    GroupMemoryBarrierWithGroupSync();

    if (!active) return;

    // Compute local rank: count how many threads with lower gid have the same digit
    uint localRank = 0;
    for (uint i = 0; i < gid; ++i) {
        if (sDigits[i] == digit) {
            localRank++;
        }
    }

    // Global output position = prefix for (digit, groupId) + local rank
    uint outIdx = sLocalOffsets[digit] + localRank;

    keysOut[outIdx] = key;
    valuesOut[outIdx] = value;
}
