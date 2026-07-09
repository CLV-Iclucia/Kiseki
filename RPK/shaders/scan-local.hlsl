// ============================================================================
// scan-local.hlsl — Per-workgroup local scan (work-efficient Blelloch)
//
// Each thread serially accumulates ITEMS_PER_THREAD elements, then the
// workgroup does a shared-memory Blelloch scan on the per-thread totals,
// and finally each thread applies its local prefix to produce final output.
//
// Elements per workgroup = WORKGROUP_SIZE * ITEMS_PER_THREAD = 256 * 16 = 4096
// ============================================================================

#include "rpk-common.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<scalar_t>   inputBuf  : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<scalar_t> outputBuf : register(u0);
[[vk::binding(2, 0)]] RWStructuredBuffer<scalar_t> blockSums : register(u1);

struct PushParams {
    uint numElements;
    uint exclusive;    // 1 = exclusive, 0 = inclusive
};
[[vk::push_constant]] PushParams pc;

#define ITEMS_PER_THREAD 16
#define ELEMENTS_PER_GROUP (WORKGROUP_SIZE * ITEMS_PER_THREAD)

groupshared scalar_t sdata[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gid : SV_GroupIndex,
          uint3 groupId : SV_GroupID) {
    uint blockBase = groupId.x * ELEMENTS_PER_GROUP;

    // ---- Phase A: each thread does serial scan of its ITEMS_PER_THREAD elements ----
    scalar_t threadData[ITEMS_PER_THREAD];
    scalar_t threadTotal = OP_IDENTITY;

    uint baseIdx = blockBase + gid * ITEMS_PER_THREAD;
    for (uint i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint idx = baseIdx + i;
        scalar_t val = (idx < pc.numElements) ? inputBuf[idx] : OP_IDENTITY;
        threadTotal = rpk_op(threadTotal, val);
        threadData[i] = threadTotal;  // inclusive scan within thread's chunk
    }

    // ---- Phase B: Blelloch scan on per-thread totals in shared memory ----
    sdata[gid] = threadTotal;
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep
    uint stride = 1;
    for (uint d = WORKGROUP_SIZE >> 1; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();
        if (gid < d) {
            uint ai = stride * (2 * gid + 1) - 1;
            uint bi = stride * (2 * gid + 2) - 1;
            sdata[bi] = rpk_op(sdata[ai], sdata[bi]);
        }
        stride <<= 1;
    }

    // Save block total and set last to identity
    GroupMemoryBarrierWithGroupSync();
    if (gid == 0) {
        blockSums[groupId.x] = sdata[WORKGROUP_SIZE - 1];
        sdata[WORKGROUP_SIZE - 1] = OP_IDENTITY;
    }

    // Down-sweep
    for (uint d2 = 1; d2 < WORKGROUP_SIZE; d2 <<= 1) {
        stride >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (gid < d2) {
            uint ai = stride * (2 * gid + 1) - 1;
            uint bi = stride * (2 * gid + 2) - 1;
            scalar_t tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] = rpk_op(tmp, sdata[bi]);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // sdata[gid] now contains the exclusive prefix for this thread's chunk
    scalar_t threadPrefix = sdata[gid];

    // ---- Phase C: write output ----
    for (uint i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint idx = baseIdx + i;
        if (idx < pc.numElements) {
            if (pc.exclusive != 0) {
                // exclusive[idx] = threadPrefix + inclusive[i-1]
                // threadData[i] is inclusive within chunk, so exclusive = prefix + threadData[i-1]
                scalar_t excVal = (i == 0) ? threadPrefix : rpk_op(threadPrefix, threadData[i - 1]);
                outputBuf[idx] = excVal;
            } else {
                // inclusive[idx] = threadPrefix + threadData[i]
                outputBuf[idx] = rpk_op(threadPrefix, threadData[i]);
            }
        }
    }
}
