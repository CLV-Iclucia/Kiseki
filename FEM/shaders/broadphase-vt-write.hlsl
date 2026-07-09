// ============================================================================
// broadphase-vt-write.hlsl — VT broad phase, pass 2 (write).
// Same traversal as the count pass, but writes each accepted candidate into the
// thread's reserved range [offsets[i], offsets[i]+count[i]). Payload per pair:
//   {globalVertex, triV0, triV1, triV2}  (4 uints).
// ============================================================================
#include "bvh-traverse.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   vtOut     : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   nodeLo    : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   nodeHi    : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<int>      lch       : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<int>      rch       : register(t3);
[[vk::binding(5, 0)]] StructuredBuffer<uint>     sortedIdx : register(t4);
[[vk::binding(6, 0)]] StructuredBuffer<double>   qLo       : register(t5);
[[vk::binding(7, 0)]] StructuredBuffer<double>   qHi       : register(t6);
[[vk::binding(8, 0)]] StructuredBuffer<uint>     triConn   : register(t7);
[[vk::binding(9, 0)]] StructuredBuffer<double>   dHatBuf   : register(t8);
[[vk::binding(10,0)]] StructuredBuffer<uint>     offsets   : register(t9);

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
    uint w = offsets[i];
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
                if (!(v0 == i || v1 == i || v2 == i)) {
                    vtOut[w * 4 + 0] = i;
                    vtOut[w * 4 + 1] = v0;
                    vtOut[w * 4 + 2] = v1;
                    vtOut[w * 4 + 3] = v2;
                    w++;
                }
            } else {
                stack[top++] = rch[node];
                node = lch[node];
                continue;
            }
        }
        if (top == 0) break;
        node = stack[--top];
    }
}
