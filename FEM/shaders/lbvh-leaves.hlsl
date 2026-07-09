// ============================================================================
// lbvh-leaves.hlsl — copy each sorted primitive's AABB into its leaf node.
// Leaf node index = (nPrs - 1 + i) for sorted position i (Karras layout).
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> nodeLo    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<double> nodeHi    : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<uint>     sortedIdx : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<double>   aabbLo    : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<double>   aabbHi    : register(t2);

struct PC { uint n; };   // n = nPrs
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;
    uint node = pc.n - 1u + i;     // leaf node id
    uint p    = sortedIdx[i];      // original primitive index
    store_double3(nodeLo, node, load_double3(aabbLo, p));
    store_double3(nodeHi, node, load_double3(aabbHi, p));
}
