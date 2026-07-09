// ============================================================================
// grad-reduce.hlsl — segment-reduce row-sorted vec3 gradient entries into the
// dense per-vertex gradient, atomic-free (one thread per row-segment).
//
// Each segment covers all entries sharing one vertex-block row (the segments
// were built by bcoo-flag + scan + bcoo-segstart over the sorted row keys), so
// distinct segments map to distinct destination rows — no two threads touch the
// same g[row], hence the read-add-write is race-free without atomics.
//
// g is read-modify-write: the barrier contribution is ADDED on top of whatever
// is already there (e.g. the elastic gradient), so the result is the full
// Newton RHS source g = g_elastic + g_barrier.
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> g         : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   valSorted : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>     rowSorted : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint>     segStart  : register(t2);

struct PC { uint numSeg; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.numSeg) return;

    uint start = segStart[s];
    uint end   = segStart[s + 1];
    uint row   = rowSorted[start];

    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (uint k = start; k < end; ++k) {
        double3 value = load_double3(valSorted, k);
        a0 += value.x;
        a1 += value.y;
        a2 += value.z;
    }
    double3 result = load_double3(g, row) + double3(a0, a1, a2);
    store_double3(g, row, result);
}
