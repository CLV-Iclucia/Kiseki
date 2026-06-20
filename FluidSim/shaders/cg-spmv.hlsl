// ============================================================================
// cg-spmv.hlsl — CG: Sparse matrix-vector multiply (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> output      : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   Adiag       : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<float>   Aneighbour0 : register(t1); // -X
[[vk::binding(3, 0)]] StructuredBuffer<float>   Aneighbour1 : register(t2); // +X
[[vk::binding(4, 0)]] StructuredBuffer<float>   Aneighbour2 : register(t3); // -Y
[[vk::binding(5, 0)]] StructuredBuffer<float>   Aneighbour3 : register(t4); // +Y
[[vk::binding(6, 0)]] StructuredBuffer<float>   Aneighbour4 : register(t5); // -Z
[[vk::binding(7, 0)]] StructuredBuffer<float>   Aneighbour5 : register(t6); // +Z
[[vk::binding(8, 0)]] StructuredBuffer<float>   src         : register(t7);

struct PushParams {
    uint3 gridSize;
    uint  numCells;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numCells) return;

    uint3 gs = pc.gridSize;
    uint3 c;
    c.x = idx % gs.x;
    c.y = (idx / gs.x) % gs.y;
    c.z = idx / (gs.x * gs.y);

    float result = Adiag[idx] * src[idx];

    if (c.x > 0)        result += Aneighbour0[idx] * src[idxCell(uint3(c.x-1, c.y, c.z), gs)];
    if (c.x+1 < gs.x)   result += Aneighbour1[idx] * src[idxCell(uint3(c.x+1, c.y, c.z), gs)];
    if (c.y > 0)        result += Aneighbour2[idx] * src[idxCell(uint3(c.x, c.y-1, c.z), gs)];
    if (c.y+1 < gs.y)   result += Aneighbour3[idx] * src[idxCell(uint3(c.x, c.y+1, c.z), gs)];
    if (c.z > 0)        result += Aneighbour4[idx] * src[idxCell(uint3(c.x, c.y, c.z-1), gs)];
    if (c.z+1 < gs.z)   result += Aneighbour5[idx] * src[idxCell(uint3(c.x, c.y, c.z+1), gs)];

    output[idx] = result;
}
