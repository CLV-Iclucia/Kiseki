// ============================================================================
// bcoo-segstart.hlsl — scatter compact row-segment start offsets.
//   segId = inclusive prefix sum of the segment-start flags (1-based).
//   For every entry that starts a new segment, write its index k into
//   segStart[segId[k]-1]. The very last entry writes the sentinel
//   segStart[numSeg] = n. The result equals CPU BlockSparseMatrix::m_rowSegments
//   plus a trailing sentinel, i.e. segStart has numSeg+1 valid entries.
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> segStart : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<uint>   flag     : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>   segId    : register(t1);

struct PC { uint n; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint k = tid.x;
    if (k >= pc.n) return;
    if (flag[k] == 1u) segStart[segId[k] - 1u] = k;
    if (k == pc.n - 1u) segStart[segId[k]] = pc.n;   // sentinel = nnz
}
