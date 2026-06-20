// ============================================================================
// g2p-gather.hlsl — Grid-to-Particle gather (trilinear) + FLIP blend
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] StructuredBuffer<float3>      positions  : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float3>    velocities : register(u0);
[[vk::binding(2, 0)]] StructuredBuffer<float>       uGrid      : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<float>       vGrid      : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<float>       wGrid      : register(t3);

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float3 origin;
    uint   numParticles;
    float  flipBlend;
};
[[vk::push_constant]] PushParams pc;

// Sample one MAC velocity component at world position.
// faceRes: size of that component's face array
// faceOffset: offset from cell origin to face centre
float sampleFace(StructuredBuffer<float> field, float3 worldPos,
                 uint3 faceRes, float3 faceOffset)
{
    float3 local = (worldPos - pc.origin - faceOffset) / pc.gridSpacing;
    int3   i0    = clamp(int3(floor(local)), int3(0,0,0), int3(faceRes) - 2);
    int3   i1    = i0 + 1;
    float3 f     = saturate(local - float3(i0));

    uint rx = faceRes.x, ry = faceRes.y;
    float c000 = field[i0.x + i0.y*rx + i0.z*rx*ry];
    float c100 = field[i1.x + i0.y*rx + i0.z*rx*ry];
    float c010 = field[i0.x + i1.y*rx + i0.z*rx*ry];
    float c110 = field[i1.x + i1.y*rx + i0.z*rx*ry];
    float c001 = field[i0.x + i0.y*rx + i1.z*rx*ry];
    float c101 = field[i1.x + i0.y*rx + i1.z*rx*ry];
    float c011 = field[i0.x + i1.y*rx + i1.z*rx*ry];
    float c111 = field[i1.x + i1.y*rx + i1.z*rx*ry];

    return lerp(lerp(lerp(c000,c100,f.x), lerp(c010,c110,f.x), f.y),
                lerp(lerp(c001,c101,f.x), lerp(c011,c111,f.x), f.y), f.z);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numParticles) return;
    float3 pos = positions[tid.x];
    float3 vel = velocities[tid.x];

    float dx   = pc.gridSpacing;
    uint3 gs   = pc.gridSize;

    uint3 uRes = uint3(gs.x + 1, gs.y,     gs.z    );
    uint3 vRes = uint3(gs.x,     gs.y + 1, gs.z    );
    uint3 wRes = uint3(gs.x,     gs.y,     gs.z + 1);

    float3 uOff = float3(0.0f,   0.5f*dx, 0.5f*dx);
    float3 vOff = float3(0.5f*dx, 0.0f,  0.5f*dx);
    float3 wOff = float3(0.5f*dx, 0.5f*dx, 0.0f  );

    float3 gridVel = float3(
        sampleFace(uGrid, pos, uRes, uOff),
        sampleFace(vGrid, pos, vRes, vOff),
        sampleFace(wGrid, pos, wRes, wOff)
    );

    // FLIP blend: newVel = lerp(picVel, oldVel + (gridVel - oldGridVel), blend)
    // Simplified (no oldGrid stored): blend toward gridVel with (1-blend)
    velocities[tid.x] = lerp(gridVel, vel, pc.flipBlend);
}
