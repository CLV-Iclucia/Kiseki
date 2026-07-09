// ============================================================================
// advect.hlsl — RK2 particle advection + collider SDF projection (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float>    positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float>    velocities;
[[vk::binding(2, 0)]] StructuredBuffer<float>      uGrid;
[[vk::binding(3, 0)]] StructuredBuffer<float>      vGrid;
[[vk::binding(4, 0)]] StructuredBuffer<float>      wGrid;
[[vk::binding(5, 0)]] Texture3D<float>             colliderSdf;
[[vk::binding(6, 0)]] SamplerState                 sdfSampler;

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float3 origin;
    uint   numParticles;
    float  dt;
};
[[vk::push_constant]] PushParams pc;

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

float3 sampleVelocity(float3 pos) {
    float  dx  = pc.gridSpacing;
    uint3  gs  = pc.gridSize;
    uint3 uRes = uint3(gs.x+1, gs.y,   gs.z  );
    uint3 vRes = uint3(gs.x,   gs.y+1, gs.z  );
    uint3 wRes = uint3(gs.x,   gs.y,   gs.z+1);
    return float3(
        sampleFace(uGrid, pos, uRes, float3(0,      0.5f*dx, 0.5f*dx)),
        sampleFace(vGrid, pos, vRes, float3(0.5f*dx, 0,      0.5f*dx)),
        sampleFace(wGrid, pos, wRes, float3(0.5f*dx, 0.5f*dx, 0     ))
    );
}

float sampleColliderSdf(float3 worldPos) {
    float3 uvw = (worldPos - pc.origin) / (float3(pc.gridSize) * pc.gridSpacing);
    return colliderSdf.SampleLevel(sdfSampler, uvw, 0);
}

// Clamp position to simulation domain
float3 clampToDomain(float3 pos) {
    float3 lo = pc.origin + 0.5f * pc.gridSpacing;
    float3 hi = pc.origin + (float3(pc.gridSize) - 0.5f) * pc.gridSpacing;
    return clamp(pos, lo, hi);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.numParticles) return;
    float3 pos = load_float3(positions, tid.x);
    float3 vel = load_float3(velocities, tid.x);

    float dt = pc.dt;

    // RK2 semi-Lagrangian advection
    float3 v0     = sampleVelocity(pos);
    float3 midPos = clampToDomain(pos + 0.5f * dt * v0);
    float3 v1     = sampleVelocity(midPos);
    float3 newPos = clampToDomain(pos + dt * v1);

    // Collider SDF projection: push particle outside if inside collider
    float sdf = sampleColliderSdf(newPos);
    if (sdf < 0.0f) {
        // Finite-difference gradient of collider SDF
        float eps = pc.gridSpacing * 0.1f;
        float3 grad;
        grad.x = sampleColliderSdf(newPos + float3(eps, 0, 0)) - sdf;
        grad.y = sampleColliderSdf(newPos + float3(0, eps, 0)) - sdf;
        grad.z = sampleColliderSdf(newPos + float3(0, 0, eps)) - sdf;
        float lenGrad = length(grad);
        if (lenGrad > 1e-6f) {
            grad /= lenGrad;
            newPos -= grad * sdf;   // push outward by |sdf|
            // Zero out the velocity component into the collider
            vel -= min(0.0f, dot(vel, grad)) * grad;
        }
    }

    store_float3(positions, tid.x, newPos);
    store_float3(velocities, tid.x, vel);
}
