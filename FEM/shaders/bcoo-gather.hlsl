// ============================================================================
// bcoo-gather.hlsl — apply a permutation to BCOO blocks + col indices.
//   blocksOut[k] = blocksIn[perm[k]]   (9 doubles, column-major dmat3)
//   colOut[k]    = colIn[perm[k]]
// The row array is sorted in place by the radix sort, so it is not gathered
// here. Output must be a distinct buffer from input (permutation is arbitrary).
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> blocksOut : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   colOut    : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<double>   blocksIn  : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<uint>     colIn     : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<uint>     perm      : register(t2);

struct PC { uint n; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint k = tid.x;
    if (k >= pc.n) return;
    uint src = perm[k];
    colOut[k] = colIn[src];
    store_double3x3_col_major(
        blocksOut, k, load_double3x3_col_major(blocksIn, src));
}
