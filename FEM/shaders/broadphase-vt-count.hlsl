// ============================================================================
// broadphase-vt-count.hlsl — VT broad phase, pass 1 (count).
// One thread per query vertex. Traverse the triangle LBVH; for each leaf whose
// raw trajectory AABB overlaps the vertex's dHat-dilated trajectory AABB and
// that does NOT contain the vertex, count a candidate. Mirrors CPU
// computeVertexTriangleCollisionPairs.
// ============================================================================
#include "bvh-traverse.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   counts    : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   nodeLo    : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   nodeHi    : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<int>      lch       : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<int>      rch       : register(t3);
[[vk::binding(5, 0)]] StructuredBuffer<uint>     sortedIdx : register(t4);
[[vk::binding(6, 0)]] StructuredBuffer<double>   qLo       : register(t5);
[[vk::binding(7, 0)]] StructuredBuffer<double>   qHi       : register(t6);
[[vk::binding(8, 0)]] StructuredBuffer<uint>     triConn   : register(t7);
[[vk::binding(9, 0)]] StructuredBuffer<double>   dHatBuf   : register(t8);

struct PC { uint numQueries; uint numPrims; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.numQueries) return;

    double dh = dHatBuf[0];
    double3 qlo = load_double3(qLo, i) - dh;
    double3 qhi = load_double3(qHi, i) + dh;

    int leafBase = (int)pc.numPrims - 1;
    uint cnt = 0;
    int stack[64];
    int top = 0;
    int node = 0;
    while (true) {
        double3 nlo = load_double3(nodeLo, (uint)node);
        double3 nhi = load_double3(nodeHi, (uint)node);
        if (boxOverlap(qlo, qhi, nlo, nhi)) {
            if (node >= leafBase) {
                uint tri = sortedIdx[node - leafBase];
                uint v0 = triConn[tri * 3 + 0], v1 = triConn[tri * 3 + 1], v2 = triConn[tri * 3 + 2];
                if (!(v0 == i || v1 == i || v2 == i)) cnt++;
            } else {
                stack[top++] = rch[node];
                node = lch[node];
                continue;
            }
        }
        if (top == 0) break;
        node = stack[--top];
    }
    counts[i] = cnt;
}
