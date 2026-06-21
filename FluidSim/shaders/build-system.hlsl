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
[[vk::binding(15, 0)]] StructuredBuffer<uint>    uValid       : register(t6);
[[vk::binding(16, 0)]] StructuredBuffer<uint>    vValid       : register(t7);
[[vk::binding(17, 0)]] StructuredBuffer<uint>    wValid       : register(t8);

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

    // Fluid cell test: cell is fluid if any of its 6 adjacent faces are valid
    bool isFluid = false;
    // U faces at (c.x, c.y, c.z) and (c.x+1, c.y, c.z)
    uint uIdx0 = c.x + c.y * (gs.x+1) + c.z * (gs.x+1) * gs.y;
    uint uIdx1 = (c.x+1) + c.y * (gs.x+1) + c.z * (gs.x+1) * gs.y;
    if (uValid[uIdx0] != 0u || uValid[uIdx1] != 0u) isFluid = true;
    // V faces at (c.x, c.y, c.z) and (c.x, c.y+1, c.z)
    uint vIdx0 = c.x + c.y * gs.x + c.z * (gs.y+1) * gs.x;
    uint vIdx1 = c.x + (c.y+1) * gs.x + c.z * (gs.y+1) * gs.x;
    if (vValid[vIdx0] != 0u || vValid[vIdx1] != 0u) isFluid = true;
    // W faces at (c.x, c.y, c.z) and (c.x, c.y, c.z+1)
    uint wIdx0 = c.x + c.y * gs.x + c.z * gs.x * gs.y;
    uint wIdx1 = c.x + c.y * gs.x + (c.z+1) * gs.x * gs.y;
    if (wValid[wIdx0] != 0u || wValid[wIdx1] != 0u) isFluid = true;
    active[idx] = isFluid ? 1u : 0u;

    if (!isFluid) {
        Adiag[idx] = 1.0f;  // identity for non-fluid cells → p = 0
        Aneighbour0[idx] = Aneighbour1[idx] =
        Aneighbour2[idx] = Aneighbour3[idx] =
        Aneighbour4[idx] = Aneighbour5[idx] = 0.0f;
        rhs[idx] = 0.0f;
        return;
    }

    // Following CPU convention: factor = dt/dx, solve Ap = rhs where rhs = weighted negative divergence
    float factor = pc.dt / pc.gridSpacing;
    float diag   = 0.0f;

    // Face indices
    uint uL = idxUFace(c, gs);
    uint uR = idxUFace(uint3(c.x+1, c.y, c.z), gs);
    uint vB = idxVFace(c, gs);
    uint vT = idxVFace(uint3(c.x, c.y+1, c.z), gs);
    uint wBk = idxWFace(c, gs);
    uint wFr = idxWFace(uint3(c.x, c.y, c.z+1), gs);

    float wuL = faceWeightsU[uL];
    float wuR = faceWeightsU[uR];
    float wvB = faceWeightsV[vB];
    float wvT = faceWeightsV[vT];
    float wwBk = faceWeightsW[wBk];
    float wwFr = faceWeightsW[wFr];

    // RHS = weighted negative divergence (CPU convention: rhs = w_L*u_L - w_R*u_R + ...)
    float rhsVal = wuL * uGrid[uL] - wuR * uGrid[uR]
                 + wvB * vGrid[vB] - wvT * vGrid[vT]
                 + wwBk * wGrid[wBk] - wwFr * wGrid[wFr];

    // Matrix: Adiag += w*factor per face, Aneighbour = -w*factor (only if neighbour exists)
    // -X
    Aneighbour0[idx] = (c.x > 0) ? -wuL * factor : 0.0f;
    diag += wuL * factor;
    // +X
    Aneighbour1[idx] = (c.x + 1 < gs.x) ? -wuR * factor : 0.0f;
    diag += wuR * factor;
    // -Y
    Aneighbour2[idx] = (c.y > 0) ? -wvB * factor : 0.0f;
    diag += wvB * factor;
    // +Y
    Aneighbour3[idx] = (c.y + 1 < gs.y) ? -wvT * factor : 0.0f;
    diag += wvT * factor;
    // -Z
    Aneighbour4[idx] = (c.z > 0) ? -wwBk * factor : 0.0f;
    diag += wwBk * factor;
    // +Z
    Aneighbour5[idx] = (c.z + 1 < gs.z) ? -wwFr * factor : 0.0f;
    diag += wwFr * factor;

    Adiag[idx] = (diag > 0.0f) ? diag : 1.0f;
    rhs[idx] = rhsVal;
}
