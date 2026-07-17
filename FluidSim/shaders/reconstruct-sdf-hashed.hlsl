// ============================================================================
// reconstruct-sdf-hashed.hlsl
// Reconstruct fluid SDF from particles using a sorted uniform-grid hash.
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWTexture3D<float> fluidSdf : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float> positions : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint> particleIndices : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint> cellStart : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<uint> cellEnd : register(t3);

struct PushParams {
    uint gridSizeX;
    uint gridSizeY;
    uint gridSizeZ;
    float gridSpacing;
    float originX;
    float originY;
    float originZ;
    uint searchRadius;
    float particleRadius;
};
[[vk::push_constant]] PushParams pc;

static const uint EMPTY_CELL = 0xffffffffu;

uint flatten(uint3 c)
{
    return c.x + c.y * pc.gridSizeX + c.z * pc.gridSizeX * pc.gridSizeY;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint nc = pc.gridSizeX * pc.gridSizeY * pc.gridSizeZ;
    if (idx >= nc) {
        return;
    }

    uint3 c;
    c.x = idx % pc.gridSizeX;
    c.y = (idx / pc.gridSizeX) % pc.gridSizeY;
    c.z = idx / (pc.gridSizeX * pc.gridSizeY);

    float3 origin = float3(pc.originX, pc.originY, pc.originZ);
    float3 cellWorld = origin + (float3(c) + 0.5f) * pc.gridSpacing;
    float minDist = pc.gridSpacing * float(pc.searchRadius + 2) + pc.particleRadius;

    int radius = int(pc.searchRadius);
    int3 base = int3(c);
    for (int dz = -radius; dz <= radius; ++dz) {
        int z = base.z + dz;
        if (z < 0 || z >= int(pc.gridSizeZ)) continue;
        for (int dy = -radius; dy <= radius; ++dy) {
            int y = base.y + dy;
            if (y < 0 || y >= int(pc.gridSizeY)) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                int x = base.x + dx;
                if (x < 0 || x >= int(pc.gridSizeX)) continue;

                uint key = flatten(uint3(x, y, z));
                uint begin = cellStart[key];
                if (begin == EMPTY_CELL) {
                    continue;
                }
                uint end = cellEnd[key];
                for (uint s = begin; s < end; ++s) {
                    uint particle = particleIndices[s];
                    float3 pos = load_float3(positions, particle);
                    float dist = length(cellWorld - pos) - pc.particleRadius;
                    minDist = min(minDist, dist);
                }
            }
        }
    }

    fluidSdf[c] = minDist;
}
