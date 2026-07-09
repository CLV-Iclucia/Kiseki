// ============================================================================
// bcoo-flag.hlsl — mark row-segment starts in a row-sorted BCOO.
//   flag[k] = 1 if k==0 or row[k] != row[k-1], else 0.
// An inclusive prefix sum of flag then gives the 1-based segment id of each
// entry, and the total segment count = scan[n-1].
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> flag : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<uint>   row  : register(t0);

struct PC { uint n; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint k = tid.x;
    if (k >= pc.n) return;
    flag[k] = (k == 0u || row[k] != row[k - 1u]) ? 1u : 0u;
}
