// ============================================================================
// build-particle-hash.hlsl
// Build sorted-grid keys for particle spatial hashing.
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> particleCellKeys : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> particleIndices : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float> positions : register(t0);

struct PushParams {
    uint gridSizeX;
    uint gridSizeY;
    uint gridSizeZ;
    float gridSpacing;
    float originX;
    float originY;
    float originZ;
    uint numParticles;
};
[[vk::push_constant]] PushParams pc;

uint flatten(uint3 c)
{
    return c.x + c.y * pc.gridSizeX + c.z * pc.gridSizeX * pc.gridSizeY;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.numParticles) {
        return;
    }

    float3 p = load_float3(positions, i);
    float3 g = floor((p - float3(pc.originX, pc.originY, pc.originZ)) / pc.gridSpacing);
    uint3 c;
    c.x = uint(clamp(int(g.x), 0, int(pc.gridSizeX) - 1));
    c.y = uint(clamp(int(g.y), 0, int(pc.gridSizeY) - 1));
    c.z = uint(clamp(int(g.z), 0, int(pc.gridSizeZ) - 1));

    particleCellKeys[i] = flatten(c);
    particleIndices[i] = i;
}
