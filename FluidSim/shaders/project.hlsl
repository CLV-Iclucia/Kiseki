// ============================================================================
// project.hlsl — Apply pressure gradient to face velocities (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> uGrid        : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> vGrid        : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> wGrid        : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>   pressure     : register(t0);
[[vk::binding(4, 0)]] StructuredBuffer<float>   faceWeightsU : register(t1);
[[vk::binding(5, 0)]] StructuredBuffer<float>   faceWeightsV : register(t2);
[[vk::binding(6, 0)]] StructuredBuffer<float>   faceWeightsW : register(t3);

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float  density;
    float  dt;
    uint   maxFaces;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.maxFaces) return;

    float scale = pc.dt / (pc.density * pc.gridSpacing);
    uint3 gs = pc.gridSize;

    // U faces
    uint nu = (gs.x + 1) * gs.y * gs.z;
    if (idx < nu) {
        float fw = faceWeightsU[idx];
        if (fw > 0.0f) {
            uint x = idx % (gs.x + 1);
            uint yz = idx / (gs.x + 1);
            uint y = yz % gs.y;
            uint z = yz / gs.y;
            float pR = (x < gs.x) ? pressure[idxCell(uint3(x,   y, z), gs)] : 0.0f;
            float pL = (x > 0)    ? pressure[idxCell(uint3(x-1, y, z), gs)] : 0.0f;
            uGrid[idx] -= scale * (pR - pL) * fw;
        }
        return;
    }
    idx -= nu;

    // V faces
    uint nv = gs.x * (gs.y + 1) * gs.z;
    if (idx < nv) {
        float fw = faceWeightsV[idx];
        if (fw > 0.0f) {
            uint y = idx % (gs.y + 1);
            uint xz = idx / (gs.y + 1);
            uint x = xz % gs.x;
            uint z = xz / gs.x;
            float pR = (y < gs.y) ? pressure[idxCell(uint3(x, y,   z), gs)] : 0.0f;
            float pL = (y > 0)    ? pressure[idxCell(uint3(x, y-1, z), gs)] : 0.0f;
            vGrid[idx] -= scale * (pR - pL) * fw;
        }
        return;
    }
    idx -= nv;

    // W faces
    {
        float fw = faceWeightsW[idx];
        if (fw > 0.0f) {
            uint x = idx % gs.x;
            uint y = (idx / gs.x) % gs.y;
            uint z = idx / (gs.x * gs.y);
            float pR = (z < gs.z) ? pressure[idxCell(uint3(x, y, z),   gs)] : 0.0f;
            float pL = (z > 0)    ? pressure[idxCell(uint3(x, y, z-1), gs)] : 0.0f;
            wGrid[idx] -= scale * (pR - pL) * fw;
        }
    }
}
