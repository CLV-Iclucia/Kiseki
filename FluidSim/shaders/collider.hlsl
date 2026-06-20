// ============================================================================
// collider.hlsl — Zero out normal velocity inside collider SDF (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> uGrid       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> vGrid       : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> wGrid       : register(u2);
[[vk::binding(3, 0)]] Texture3D<float>          colliderSdf : register(t0);
[[vk::binding(4, 0)]] SamplerState              sdfSampler  : register(s0);

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float3 origin;
    uint   maxFaces;
};
[[vk::push_constant]] PushParams pc;

float sampleCollider(float3 world) {
    float3 uvw = (world - pc.origin) / (float3(pc.gridSize) * pc.gridSpacing);
    return colliderSdf.SampleLevel(sdfSampler, uvw, 0);
}

void zeroNormalComponent(RWStructuredBuffer<float> field, uint idx,
                         float3 pos, float gradComp)
{
    float sdf = sampleCollider(pos);
    if (sdf < 0.0f) {
        field[idx] *= (1.0f - abs(gradComp));
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.maxFaces) return;

    float dx   = pc.gridSpacing;
    float eps  = dx * 0.01f;
    uint3 gs   = pc.gridSize;

    // U faces
    uint nu = (gs.x + 1) * gs.y * gs.z;
    if (idx < nu) {
        uint x = idx % (gs.x + 1);
        uint yz = idx / (gs.x + 1);
        uint y = yz % gs.y;
        uint z = yz / gs.y;
        if (x == 0 || x == gs.x) return;
        float3 pos = pc.origin + float3(x, y + 0.5f, z + 0.5f) * dx;
        float sdf = sampleCollider(pos);
        if (sdf < 0.0f) {
            float gx = (sampleCollider(pos + float3(eps,0,0)) - sdf) / eps;
            float gy = (sampleCollider(pos + float3(0,eps,0)) - sdf) / eps;
            float gz = (sampleCollider(pos + float3(0,0,eps)) - sdf) / eps;
            float lenG = length(float3(gx,gy,gz));
            if (lenG > 1e-6f) { gx /= lenG; }
            uGrid[idx] *= (1.0f - abs(gx));
        }
        return;
    }
    idx -= nu;

    // V faces
    uint nv = gs.x * (gs.y + 1) * gs.z;
    if (idx < nv) {
        uint y = idx % (gs.y + 1);
        uint xz = idx / (gs.y + 1);
        uint x = xz % gs.x;
        uint z = xz / gs.x;
        if (y == 0 || y == gs.y) return;
        float3 pos = pc.origin + float3(x + 0.5f, y, z + 0.5f) * dx;
        float sdf = sampleCollider(pos);
        if (sdf < 0.0f) {
            float gx = (sampleCollider(pos + float3(eps,0,0)) - sdf) / eps;
            float gy = (sampleCollider(pos + float3(0,eps,0)) - sdf) / eps;
            float gz = (sampleCollider(pos + float3(0,0,eps)) - sdf) / eps;
            float lenG = length(float3(gx,gy,gz));
            if (lenG > 1e-6f) { gy /= lenG; }
            vGrid[idx] *= (1.0f - abs(gy));
        }
        return;
    }
    idx -= nv;

    // W faces
    {
        uint x = idx % gs.x;
        uint y = (idx / gs.x) % gs.y;
        uint z = idx / (gs.x * gs.y);
        if (z == 0 || z == gs.z) return;
        float3 pos = pc.origin + float3(x + 0.5f, y + 0.5f, z) * dx;
        float sdf = sampleCollider(pos);
        if (sdf < 0.0f) {
            float gx = (sampleCollider(pos + float3(eps,0,0)) - sdf) / eps;
            float gy = (sampleCollider(pos + float3(0,eps,0)) - sdf) / eps;
            float gz = (sampleCollider(pos + float3(0,0,eps)) - sdf) / eps;
            float lenG = length(float3(gx,gy,gz));
            if (lenG > 1e-6f) { gz /= lenG; }
            wGrid[idx] *= (1.0f - abs(gz));
        }
    }
}
