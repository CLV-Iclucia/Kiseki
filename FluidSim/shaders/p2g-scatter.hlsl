// ============================================================================
// p2g-scatter.hlsl — Particle-to-Grid scatter (trilinear hat + float atomic CAS)
// Float32 throughout; atomic add via uint CAS trick.
// ============================================================================
#include "fluid-common.hlsl"

// Grid/weight buffers declared as uint for atomic CAS; read as float elsewhere.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> uGrid      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> vGrid      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> wGrid      : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> uWeights   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> vWeights   : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> wWeights   : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<float> positions  : register(t0);
[[vk::binding(7, 0)]] StructuredBuffer<float> velocities : register(t1);

struct PushParams {
    uint3 gridSize;
    float gridSpacing;
    float3 origin;
    uint  numParticles;
    float dt;
};
[[vk::push_constant]] PushParams pc;

// Splat a single velocity component to one set of staggered faces.
// faceOffset: world-space offset of face centres from cell (0,0,0) corner
//   U-faces at (i,    j+0.5, k+0.5)  → faceOffset = (0, 0.5*dx, 0.5*dx)
//   V-faces at (i+0.5, j,    k+0.5)  → faceOffset = (0.5*dx, 0, 0.5*dx)
//   W-faces at (i+0.5, j+0.5, k)     → faceOffset = (0.5*dx, 0.5*dx, 0)
void splatFace(RWStructuredBuffer<uint> field,
               RWStructuredBuffer<uint> weights,
               float3 worldPos, float value,
               uint3 faceRes, float3 faceOffset)
{
    float3 local = (worldPos - pc.origin - faceOffset) / pc.gridSpacing;
    int3   base  = int3(floor(local));
    float3 frac  = local - float3(base);

    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                int3 idx3 = base + int3(dx, dy, dz);
                if (any(idx3 < 0) || idx3.x >= int(faceRes.x) ||
                    idx3.y >= int(faceRes.y) || idx3.z >= int(faceRes.z))
                    continue;

                float wx = hatKernel(float(dx) - frac.x);
                float wy = hatKernel(float(dy) - frac.y);
                float wz = hatKernel(float(dz) - frac.z);
                float w  = wx * wy * wz;
                if (w <= 0.0f) continue;

                uint flatIdx = idx3.x
                             + idx3.y * faceRes.x
                             + idx3.z * faceRes.x * faceRes.y;
                atomicAddFloat(field,   flatIdx, w * value);
                atomicAddFloat(weights, flatIdx, w);
            }
        }
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numParticles) return;
    uint base = tid.x * 3;
    float3 pos = float3(positions[base], positions[base+1], positions[base+2]);
    float3 vel = float3(velocities[base], velocities[base+1], velocities[base+2]);

    float dx = pc.gridSpacing;
    uint3 gs = pc.gridSize;

    uint3 uRes = uint3(gs.x + 1, gs.y,     gs.z    );
    uint3 vRes = uint3(gs.x,     gs.y + 1, gs.z    );
    uint3 wRes = uint3(gs.x,     gs.y,     gs.z + 1);

    float3 uOff = float3(0.0f,   0.5f*dx, 0.5f*dx);
    float3 vOff = float3(0.5f*dx, 0.0f,  0.5f*dx);
    float3 wOff = float3(0.5f*dx, 0.5f*dx, 0.0f  );

    splatFace(uGrid, uWeights, pos, vel.x, uRes, uOff);
    splatFace(vGrid, vWeights, pos, vel.y, vRes, vOff);
    splatFace(wGrid, wWeights, pos, vel.z, wRes, wOff);
}
