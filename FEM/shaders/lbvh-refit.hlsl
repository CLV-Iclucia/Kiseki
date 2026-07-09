// ============================================================================
// lbvh-refit.hlsl — bottom-up internal-node AABB refit (atomic-flag climb).
// One thread per leaf. Each climbs toward the root; the FIRST child to reach a
// parent bails out, the SECOND computes the parent's AABB from both children.
// Mirrors spatify::LBVH refit (std::atomic exchange). nodeLo/Hi are
// globallycoherent and a DeviceMemoryBarrier publishes each parent write before
// the thread climbs further. flags[nPrs-1] must be zero-initialised.
// ============================================================================
[[vk::binding(0, 0)]] globallycoherent RWStructuredBuffer<double> nodeLo : register(u0);
[[vk::binding(1, 0)]] globallycoherent RWStructuredBuffer<double> nodeHi : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>  flags : register(u2);   // [nPrs-1]
[[vk::binding(3, 0)]] StructuredBuffer<int>     lch   : register(t0);
[[vk::binding(4, 0)]] StructuredBuffer<int>     rch   : register(t1);
[[vk::binding(5, 0)]] StructuredBuffer<int>     fa    : register(t2);

struct PC { uint n; };   // n = nPrs
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;

    int node   = (int)(pc.n - 1u + i);   // leaf
    int parent = fa[node];
    while (parent != -1) {
        uint old;
        InterlockedExchange(flags[parent], 1u, old);
        if (old == 0u) return;           // first child: leave parent to the second

        int lc = lch[parent], rc = rch[parent];
        [unroll] for (uint d = 0; d < 3u; ++d) {
            nodeLo[parent * 3 + d] = min(nodeLo[lc * 3 + d], nodeLo[rc * 3 + d]);
            nodeHi[parent * 3 + d] = max(nodeHi[lc * 3 + d], nodeHi[rc * 3 + d]);
        }
        DeviceMemoryBarrier();           // publish parent AABB before climbing

        node   = parent;
        parent = fa[node];
    }
}
