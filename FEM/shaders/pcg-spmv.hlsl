// ============================================================================
// pcg-spmv.hlsl — BCOO block SpMV (double), segment-reduce, atomic-free.
// One thread per row-segment; identical semantics to CPU sortByRow path.
// Vectors are scalar double[N*3] (no struct padding). Block: column-major 9.
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> Ap       : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   blocks   : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>     rowIdx   : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint>     colIdx   : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<uint>     segStart : register(t3);
[[vk::binding(5, 0)]] StructuredBuffer<double>   p        : register(t4);

struct PC { uint numSeg; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.numSeg) return;

    uint start = segStart[s];
    uint end   = segStart[s + 1];
    uint row   = rowIdx[start];

    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (uint k = start; k < end; ++k) {
        double3 vectorValue = load_double3(p, colIdx[k]);
        double3x3 block = load_double3x3_col_major(blocks, k);
        a0 += block[0][0] * vectorValue.x
            + block[0][1] * vectorValue.y
            + block[0][2] * vectorValue.z;
        a1 += block[1][0] * vectorValue.x
            + block[1][1] * vectorValue.y
            + block[1][2] * vectorValue.z;
        a2 += block[2][0] * vectorValue.x
            + block[2][1] * vectorValue.y
            + block[2][2] * vectorValue.z;
    }
    store_double3(Ap, row, double3(a0, a1, a2));
}
