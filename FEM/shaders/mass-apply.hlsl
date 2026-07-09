// ============================================================================
// mass-apply.hlsl — consistent (per-tet) mass matrix times a vector, y = M v.
// One thread per vertex, atomic-free gather over incident tets (same CSR-like
// adjacency as elastic-gradient.hlsl). The consistent mass block for tet t is
//   M_block(j,k) = density * restVol_t * (j==k ? 0.1 : 0.05) * I3
// (matching CPU ElasticTetMesh::assembleMassMatrix). Used for the inertial
// gradient M(x - x_hat) and inertial energy 0.5 (x-x_hat)^T M (x-x_hat).
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> y        : register(u0); // [N*3]
[[vk::binding(1, 0)]] StructuredBuffer<double>   v        : register(t0); // [N*3]
[[vk::binding(2, 0)]] StructuredBuffer<uint>     tetConn  : register(t1); // [M*4]
[[vk::binding(3, 0)]] StructuredBuffer<double>   vol      : register(t2); // [M]
[[vk::binding(4, 0)]] StructuredBuffer<uint>     adjStart : register(t3); // [N+1]
[[vk::binding(5, 0)]] StructuredBuffer<uint>     adjTet   : register(t4); // [K]
[[vk::binding(6, 0)]] StructuredBuffer<uint>     adjLocal : register(t5); // [K]

struct PC { double density; uint numVerts; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint vtx = tid.x;
    if (vtx >= pc.numVerts) return;

    double3 acc = double3(0.0, 0.0, 0.0);
    uint a0 = adjStart[vtx], a1 = adjStart[vtx + 1];
    for (uint a = a0; a < a1; ++a) {
        uint tet = adjTet[a];
        uint lv  = adjLocal[a];
        double w = pc.density * vol[tet];
        uint cb  = tet * 4u;
        [unroll] for (uint k = 0; k < 4u; ++k) {
            double coeff = w * (lv == k ? 0.1 : 0.05);
            acc += coeff * load_double3(v, tetConn[cb + k]);
        }
    }
    store_double3(y, vtx, acc);
}
