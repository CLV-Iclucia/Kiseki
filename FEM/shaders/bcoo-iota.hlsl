// ============================================================================
// bcoo-iota.hlsl — initialize a permutation buffer: values[i] = i.
// Used as the value array for the radix sort (key = row), so after sorting
// values[k] holds the original index of the entry now at position k.
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> values : register(u0);

struct PC { uint n; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.n) return;
    values[tid.x] = tid.x;
}
