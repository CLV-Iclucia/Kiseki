// ============================================================================
// radix-histogram.hlsl — Radix sort: per-workgroup digit histogram
//
// Counts occurrences of each 4-bit digit within each workgroup's tile.
// Layout: histogram[digit * numGroups + groupId.x] = count
// This column-major layout means scanning a single row gives the global
// prefix for that digit across all workgroups.
// ============================================================================

#define WORKGROUP_SIZE 256
#define NUM_BUCKETS 16

[[vk::binding(0, 0)]] StructuredBuffer<uint>   keys      : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram : register(u0);

struct PushParams {
    uint numElements;
    uint bitOffset;
    uint numGroups;
};
[[vk::push_constant]] PushParams pc;

groupshared uint localHist[NUM_BUCKETS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    // Clear local histogram
    if (gid < NUM_BUCKETS) {
        localHist[gid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Each thread extracts its digit and atomically increments local count
    if (dtid.x < pc.numElements) {
        uint key = keys[dtid.x];
        uint digit = (key >> pc.bitOffset) & 0xF;
        InterlockedAdd(localHist[digit], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Write local histogram to global (column-major)
    if (gid < NUM_BUCKETS) {
        histogram[gid * pc.numGroups + groupId.x] = localHist[gid];
    }
}
