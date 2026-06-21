// ============================================================================
// build-weights.hlsl — Compute u/v/w face open-fraction from collider SDF
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> faceWeightsU : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> faceWeightsV : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> faceWeightsW : register(u2);
[[vk::binding(3, 0)]] Texture3D<float>  colliderSdf          : register(t0);
[[vk::binding(4, 0)]] SamplerState      sdfSampler           : register(s0);

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float3 origin;
    uint   maxFaces;
};
[[vk::push_constant]] PushParams pc;

float sampleColliderSdf(float3 worldPos) {
    float3 uvw = (worldPos - pc.origin) / (float3(pc.gridSize) * pc.gridSpacing);
    float sdf = colliderSdf.SampleLevel(sdfSampler, uvw, 0);
    // If collider SDF is uninitialized (all zeros), treat as fully open
    return (sdf == 0.0f) ? 1.0f : sdf;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.maxFaces) return;

    float dx = pc.gridSpacing;
    uint3 gs = pc.gridSize;

    // U faces: boundary at x==0 and x==gs.x (domain walls)
    uint nu = (gs.x + 1) * gs.y * gs.z;
    if (idx < nu) {
        uint x = idx % (gs.x + 1);
        uint yz = idx / (gs.x + 1);
        uint y = yz % gs.y;
        uint z = yz / gs.y;
        // Domain boundary: solid wall
        float w = (x == 0 || x == gs.x) ? 0.0f : 1.0f;
        // Collider SDF (if initialized)
        if (w > 0.0f) {
            float3 wp = pc.origin + float3(x, y + 0.5f, z + 0.5f) * dx;
            float sdf = sampleColliderSdf(wp);
            if (sdf < 0.0f) w = 0.0f;
        }
        faceWeightsU[idx] = w;
        return;
    }
    idx -= nu;

    // V faces: boundary at y==0 and y==gs.y
    uint nv = gs.x * (gs.y + 1) * gs.z;
    if (idx < nv) {
        uint x = idx % gs.x;
        uint y = (idx / gs.x) % (gs.y + 1);
        uint z = idx / (gs.x * (gs.y + 1));
        float w = (y == 0 || y == gs.y) ? 0.0f : 1.0f;
        if (w > 0.0f) {
            float3 wp = pc.origin + float3(x + 0.5f, y, z + 0.5f) * dx;
            float sdf = sampleColliderSdf(wp);
            if (sdf < 0.0f) w = 0.0f;
        }
        faceWeightsV[idx] = w;
        return;
    }
    idx -= nv;

    // W faces: boundary at z==0 and z==gs.z
    {
        uint x = idx % gs.x;
        uint y = (idx / gs.x) % gs.y;
        uint z = idx / (gs.x * gs.y);
        float w = (z == 0 || z == gs.z) ? 0.0f : 1.0f;
        if (w > 0.0f) {
            float3 wp = pc.origin + float3(x + 0.5f, y + 0.5f, z) * dx;
            float sdf = sampleColliderSdf(wp);
            if (sdf < 0.0f) w = 0.0f;
        }
        faceWeightsW[idx] = w;
    }
}
