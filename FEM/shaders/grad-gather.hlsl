// ============================================================================
// grad-gather.hlsl — apply a permutation to scattered vec3 gradient values.
//   valOut[k] = valIn[perm[k]]   (3 doubles per entry)
// Companion to bcoo-iota/bcoo-flag/bcoo-segstart for the gradient scatter-add:
// after radix-sorting the row keys (value = perm), this gathers the matching
// vec3 contributions into a row-sorted scratch buffer. Output must be distinct
// from input (the permutation is arbitrary).
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> valOut : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   valIn  : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>     perm   : register(t1);

struct PC { uint n; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint k = tid.x;
    if (k >= pc.n) return;
    uint s = perm[k];
    store_double3(valOut, k, load_double3(valIn, s));
}
