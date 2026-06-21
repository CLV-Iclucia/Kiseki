// ============================================================================
// reconstruct-sdf.hlsl — Reconstruct fluid SDF from particles
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWTexture3D<float> fluidSdf  : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float> positions : register(t0);

struct PushParams {
    uint3 gridSize;
    float gridSpacing;
    float3 origin;
    uint numParticles;
    float particleRadius;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint nc = pc.gridSize.x * pc.gridSize.y * pc.gridSize.z;
    if (idx >= nc) return;

    uint3 c;
    c.x = idx % pc.gridSize.x;
    c.y = (idx / pc.gridSize.x) % pc.gridSize.y;
    c.z = idx / (pc.gridSize.x * pc.gridSize.y);

    float3 cellWorld = pc.origin + (float3(c) + 0.5f) * pc.gridSpacing;
    float minDist = 1e9f;

    // Simplified: sequential particle scan (Phase 1 approach)
    for (uint i = 0; i < pc.numParticles; ++i) {
        uint b = i * 3;
        float3 pos = float3(positions[b], positions[b+1], positions[b+2]);
        float3 d = cellWorld - pos;
        float dist = length(d) - pc.particleRadius;
        minDist = min(minDist, dist);
    }

    fluidSdf[uint3(c)] = minDist;
}
