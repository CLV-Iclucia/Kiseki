// ============================================================================
// p2g-normalize.hlsl — Normalize face velocities by accumulated weights
// Buffers are uint (because P2G uses atomic CAS); read/written as float here.
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> uGrid    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> vGrid    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> wGrid    : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<uint>   uWeights : register(t0);
[[vk::binding(4, 0)]] StructuredBuffer<uint>   vWeights : register(t1);
[[vk::binding(5, 0)]] StructuredBuffer<uint>   wWeights : register(t2);
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> uValid   : register(u3);
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> vValid   : register(u4);
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> wValid   : register(u5);

struct PushParams {
    uint3 gridSize;
    uint  maxFaces;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.maxFaces) return;

    uint3 gs = pc.gridSize;

    // U faces
    uint nu = (gs.x + 1) * gs.y * gs.z;
    if (idx < nu) {
        float w   = asfloat(uWeights[idx]);
        float vel = asfloat(uGrid[idx]);
        uGrid[idx]  = asuint(w > 0.0f ? vel / w : 0.0f);
        uValid[idx] = (w > 0.0f) ? 1u : 0u;
        return;
    }
    idx -= nu;

    // V faces
    uint nv = gs.x * (gs.y + 1) * gs.z;
    if (idx < nv) {
        float w   = asfloat(vWeights[idx]);
        float vel = asfloat(vGrid[idx]);
        vGrid[idx]  = asuint(w > 0.0f ? vel / w : 0.0f);
        vValid[idx] = (w > 0.0f) ? 1u : 0u;
        return;
    }
    idx -= nv;

    // W faces
    {
        float w   = asfloat(wWeights[idx]);
        float vel = asfloat(wGrid[idx]);
        wGrid[idx]  = asuint(w > 0.0f ? vel / w : 0.0f);
        wValid[idx] = (w > 0.0f) ? 1u : 0u;
    }
}
