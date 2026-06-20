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
    return colliderSdf.SampleLevel(sdfSampler, uvw, 0);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.maxFaces) return;

    float dx = pc.gridSpacing;
    uint3 gs = pc.gridSize;

    // U faces: centre at (i, j+0.5, k+0.5)
    uint nu = (gs.x + 1) * gs.y * gs.z;
    if (idx < nu) {
        uint x = idx % (gs.x + 1);
        uint yz = idx / (gs.x + 1);
        uint y = yz % gs.y;
        uint z = yz / gs.y;
        float3 wp = pc.origin + float3(x, y + 0.5f, z + 0.5f) * dx;
        float sdf = sampleColliderSdf(wp);
        faceWeightsU[idx] = saturate(sdf / dx);
        return;
    }
    idx -= nu;

    // V faces: centre at (i+0.5, j, k+0.5)
    uint nv = gs.x * (gs.y + 1) * gs.z;
    if (idx < nv) {
        uint y = idx % (gs.y + 1);
        uint xz = idx / (gs.y + 1);
        uint x = xz % gs.x;
        uint z = xz / gs.x;
        float3 wp = pc.origin + float3(x + 0.5f, y, z + 0.5f) * dx;
        float sdf = sampleColliderSdf(wp);
        faceWeightsV[idx] = saturate(sdf / dx);
        return;
    }
    idx -= nv;

    // W faces: centre at (i+0.5, j+0.5, k)
    {
        uint x = idx % gs.x;
        uint y = (idx / gs.x) % gs.y;
        uint z = idx / (gs.x * gs.y);
        float3 wp = pc.origin + float3(x + 0.5f, y + 0.5f, z) * dx;
        float sdf = sampleColliderSdf(wp);
        faceWeightsW[idx] = saturate(sdf / dx);
    }
}
