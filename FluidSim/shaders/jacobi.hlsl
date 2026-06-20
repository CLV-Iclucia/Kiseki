// ============================================================================
// jacobi.hlsl — Jacobi pressure iteration (one ping-pong step, float32)
//   p_out[i] = (rhs[i] - sum_j(A[i,j] * p_in[j])) / A[i,i]
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0,  0)]] StructuredBuffer<float>   pressureIn   : register(t0);
[[vk::binding(1,  0)]] RWStructuredBuffer<float> pressureOut  : register(u0);
[[vk::binding(2,  0)]] StructuredBuffer<float>   Adiag        : register(t1);
[[vk::binding(3,  0)]] StructuredBuffer<float>   Aneighbour0  : register(t2); // -X
[[vk::binding(4,  0)]] StructuredBuffer<float>   Aneighbour1  : register(t3); // +X
[[vk::binding(5,  0)]] StructuredBuffer<float>   Aneighbour2  : register(t4); // -Y
[[vk::binding(6,  0)]] StructuredBuffer<float>   Aneighbour3  : register(t5); // +Y
[[vk::binding(7,  0)]] StructuredBuffer<float>   Aneighbour4  : register(t6); // -Z
[[vk::binding(8,  0)]] StructuredBuffer<float>   Aneighbour5  : register(t7); // +Z
[[vk::binding(9,  0)]] StructuredBuffer<float>   rhs          : register(t8);
[[vk::binding(10, 0)]] StructuredBuffer<uint>    active       : register(t9);

struct PushParams {
    uint3 gridSize;
    uint  numCells;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numCells) return;

    if (active[idx] == 0u) {
        pressureOut[idx] = 0.0f;
        return;
    }

    uint3 gs = pc.gridSize;
    uint3 c;
    c.x = idx % gs.x;
    c.y = (idx / gs.x) % gs.y;
    c.z = idx / (gs.x * gs.y);

    float diag = Adiag[idx];
    float off  = rhs[idx];

    if (c.x > 0)       off -= Aneighbour0[idx] * pressureIn[idxCell(uint3(c.x-1, c.y, c.z), gs)];
    if (c.x+1 < gs.x)  off -= Aneighbour1[idx] * pressureIn[idxCell(uint3(c.x+1, c.y, c.z), gs)];
    if (c.y > 0)       off -= Aneighbour2[idx] * pressureIn[idxCell(uint3(c.x, c.y-1, c.z), gs)];
    if (c.y+1 < gs.y)  off -= Aneighbour3[idx] * pressureIn[idxCell(uint3(c.x, c.y+1, c.z), gs)];
    if (c.z > 0)       off -= Aneighbour4[idx] * pressureIn[idxCell(uint3(c.x, c.y, c.z-1), gs)];
    if (c.z+1 < gs.z)  off -= Aneighbour5[idx] * pressureIn[idxCell(uint3(c.x, c.y, c.z+1), gs)];

    pressureOut[idx] = (abs(diag) > 1e-7f) ? (off / diag) : 0.0f;
}
