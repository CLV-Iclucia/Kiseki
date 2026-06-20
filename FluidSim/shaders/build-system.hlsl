// ============================================================================
// build-system.hlsl — Build 7-point stencil Poisson system (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0,  0)]] RWStructuredBuffer<float> Adiag        : register(u0);
[[vk::binding(1,  0)]] RWStructuredBuffer<float> Aneighbour0  : register(u1); // -X
[[vk::binding(2,  0)]] RWStructuredBuffer<float> Aneighbour1  : register(u2); // +X
[[vk::binding(3,  0)]] RWStructuredBuffer<float> Aneighbour2  : register(u3); // -Y
[[vk::binding(4,  0)]] RWStructuredBuffer<float> Aneighbour3  : register(u4); // +Y
[[vk::binding(5,  0)]] RWStructuredBuffer<float> Aneighbour4  : register(u5); // -Z
[[vk::binding(6,  0)]] RWStructuredBuffer<float> Aneighbour5  : register(u6); // +Z
[[vk::binding(7,  0)]] RWStructuredBuffer<float> rhs          : register(u7);
[[vk::binding(8,  0)]] RWStructuredBuffer<uint>  active       : register(u8);

[[vk::binding(9,  0)]] StructuredBuffer<float>   uGrid        : register(t0);
[[vk::binding(10, 0)]] StructuredBuffer<float>   vGrid        : register(t1);
[[vk::binding(11, 0)]] StructuredBuffer<float>   wGrid        : register(t2);
[[vk::binding(12, 0)]] StructuredBuffer<float>   faceWeightsU : register(t3);
[[vk::binding(13, 0)]] StructuredBuffer<float>   faceWeightsV : register(t4);
[[vk::binding(14, 0)]] StructuredBuffer<float>   faceWeightsW : register(t5);
[[vk::binding(15, 0)]] Texture3D<float>          fluidSdf     : register(t6);
[[vk::binding(16, 0)]] SamplerState              sdfSampler   : register(s0);

struct PushParams {
    uint3  gridSize;
    float  gridSpacing;
    float  density;
    float  dt;
    float3 origin;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint nc = pc.gridSize.x * pc.gridSize.y * pc.gridSize.z;
    if (idx >= nc) return;

    uint3 gs = pc.gridSize;
    uint3 c;
    c.x = idx % gs.x;
    c.y = (idx / gs.x) % gs.y;
    c.z = idx / (gs.x * gs.y);

    // Fluid cell test via fluidSdf image
    float3 worldPos = pc.origin + (float3(c) + 0.5f) * pc.gridSpacing;
    float3 uvw = (worldPos - pc.origin) / (float3(gs) * pc.gridSpacing);
    float sdf = fluidSdf.SampleLevel(sdfSampler, uvw, 0);
    bool isFluid = (sdf < 0.0f);
    active[idx] = isFluid ? 1u : 0u;

    if (!isFluid) {
        Adiag[idx] = 1.0f;  // identity for non-fluid cells → p = 0
        Aneighbour0[idx] = Aneighbour1[idx] =
        Aneighbour2[idx] = Aneighbour3[idx] =
        Aneighbour4[idx] = Aneighbour5[idx] = 0.0f;
        rhs[idx] = 0.0f;
        return;
    }

    float scale = 1.0f / (pc.density * pc.gridSpacing * pc.gridSpacing);
    float diag  = 0.0f;
    float divU  = 0.0f;

    // -X face
    float fx = (c.x > 0) ? faceWeightsU[idxUFace(c, gs)] : 0.0f;
    diag += fx * scale;
    Aneighbour0[idx] = -fx * scale;
    if (c.x > 0) divU += uGrid[idxUFace(c, gs)] * fx;

    // +X face
    float fxp = (c.x + 1 < gs.x) ? faceWeightsU[idxUFace(uint3(c.x+1, c.y, c.z), gs)] : 0.0f;
    diag += fxp * scale;
    Aneighbour1[idx] = -fxp * scale;
    if (c.x + 1 < gs.x) divU -= uGrid[idxUFace(uint3(c.x+1, c.y, c.z), gs)] * fxp;

    // -Y face
    float fy = (c.y > 0) ? faceWeightsV[idxVFace(c, gs)] : 0.0f;
    diag += fy * scale;
    Aneighbour2[idx] = -fy * scale;
    if (c.y > 0) divU += vGrid[idxVFace(c, gs)] * fy;

    // +Y face
    float fyp = (c.y + 1 < gs.y) ? faceWeightsV[idxVFace(uint3(c.x, c.y+1, c.z), gs)] : 0.0f;
    diag += fyp * scale;
    Aneighbour3[idx] = -fyp * scale;
    if (c.y + 1 < gs.y) divU -= vGrid[idxVFace(uint3(c.x, c.y+1, c.z), gs)] * fyp;

    // -Z face
    float fz = (c.z > 0) ? faceWeightsW[idxWFace(c, gs)] : 0.0f;
    diag += fz * scale;
    Aneighbour4[idx] = -fz * scale;
    if (c.z > 0) divU += wGrid[idxWFace(c, gs)] * fz;

    // +Z face
    float fzp = (c.z + 1 < gs.z) ? faceWeightsW[idxWFace(uint3(c.x, c.y, c.z+1), gs)] : 0.0f;
    diag += fzp * scale;
    Aneighbour5[idx] = -fzp * scale;
    if (c.z + 1 < gs.z) divU -= wGrid[idxWFace(uint3(c.x, c.y, c.z+1), gs)] * fzp;

    Adiag[idx] = (diag > 0.0f) ? diag : 1.0f;  // avoid zero diagonal

    // RHS = -div(u) / dt  (density absorbed into scale above)
    divU /= pc.gridSpacing;
    rhs[idx] = -divU / pc.dt;
}
