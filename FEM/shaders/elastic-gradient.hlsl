// ============================================================================
// elastic-gradient.hlsl — Stable Neo-Hookean elastic energy gradient (double).
//
// Gather-per-vertex (atomic-free): one thread per vertex sums contributions
// from all incident tets. PK1 needs only first derivatives (no SVD).
//
// Per tet: F = Ds * DmInv,  J = det F,  cof = cofactor(F)
//          P = mu*F + (lambda*(J-1) - mu) * cof          (PK1)
// Per-vertex force contribution (matches PFPx^T vec(P) * vol):
//   local vertex k>=1: f = P * DmInv.row(k-1)
//   local vertex 0   : f = -(P * (row0+row1+row2))
//
// Matrices (DmInv) are stored column-major: M(r,c) = DmInv[tet*9 + c*3 + r].
// Vectors are scalar double[N*3].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> grad     : register(u0);  // [N*3]
[[vk::binding(1, 0)]] StructuredBuffer<double>   x        : register(t0);  // [N*3]
[[vk::binding(2, 0)]] StructuredBuffer<uint>     tetConn  : register(t1);  // [M*4]
[[vk::binding(3, 0)]] StructuredBuffer<double>   DmInv    : register(t2);  // [M*9]
[[vk::binding(4, 0)]] StructuredBuffer<double>   vol      : register(t3);  // [M]
[[vk::binding(5, 0)]] StructuredBuffer<uint>     adjStart : register(t4);  // [N+1]
[[vk::binding(6, 0)]] StructuredBuffer<uint>     adjTet   : register(t5);  // [K]
[[vk::binding(7, 0)]] StructuredBuffer<uint>     adjLocal : register(t6);  // [K]
[[vk::binding(8, 0)]] StructuredBuffer<double>   material : register(t7);  // [mu, lambda]

struct PC { uint numVerts; };
[[vk::push_constant]] PC pc;

double3 ld(uint i) { return load_double3(x, i); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint v = tid.x;
    if (v >= pc.numVerts) return;

    double mu = material[0], lambda = material[1];
    double3 g = double3(0.0, 0.0, 0.0);

    uint a0 = adjStart[v], a1 = adjStart[v + 1];
    for (uint a = a0; a < a1; ++a) {
        uint tet = adjTet[a];
        uint lv  = adjLocal[a];
        uint cb  = tet * 4u;
        double3x3 D = load_double3x3_col_major(DmInv, tet);

        double3 p0 = ld(tetConn[cb + 0]);
        double3 e0 = ld(tetConn[cb + 1]) - p0;
        double3 e1 = ld(tetConn[cb + 2]) - p0;
        double3 e2 = ld(tetConn[cb + 3]) - p0;

        // F(:,c) = e0*DmInv(0,c) + e1*DmInv(1,c) + e2*DmInv(2,c)
        double F[3][3];
        [unroll] for (uint c = 0; c < 3; ++c) {
            double m0 = D[0][c];
            double m1 = D[1][c];
            double m2 = D[2][c];
            F[0][c] = e0.x * m0 + e1.x * m1 + e2.x * m2;
            F[1][c] = e0.y * m0 + e1.y * m1 + e2.y * m2;
            F[2][c] = e0.z * m0 + e1.z * m1 + e2.z * m2;
        }

        double det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
                   - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
                   + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

        // cofactor(F): cof(i,j) = d det / d F(i,j)
        double cof[3][3];
        cof[0][0] = F[1][1] * F[2][2] - F[1][2] * F[2][1];
        cof[0][1] = F[1][2] * F[2][0] - F[1][0] * F[2][2];
        cof[0][2] = F[1][0] * F[2][1] - F[1][1] * F[2][0];
        cof[1][0] = F[0][2] * F[2][1] - F[0][1] * F[2][2];
        cof[1][1] = F[0][0] * F[2][2] - F[0][2] * F[2][0];
        cof[1][2] = F[0][1] * F[2][0] - F[0][0] * F[2][1];
        cof[2][0] = F[0][1] * F[1][2] - F[0][2] * F[1][1];
        cof[2][1] = F[0][2] * F[1][0] - F[0][0] * F[1][2];
        cof[2][2] = F[0][0] * F[1][1] - F[0][1] * F[1][0];

        double coef = lambda * (det - 1.0) - mu;
        double P[3][3];
        [unroll] for (uint r = 0; r < 3; ++r)
            [unroll] for (uint c = 0; c < 3; ++c)
                P[r][c] = mu * F[r][c] + coef * cof[r][c];

        // d = the DmInv row(s) that this local vertex multiplies.
        double d0, d1, d2;
        if (lv == 0u) {
            // d = -(row0 + row1 + row2) = negative column sums of DmInv
            d0 = -(D[0][0] + D[1][0] + D[2][0]);
            d1 = -(D[0][1] + D[1][1] + D[2][1]);
            d2 = -(D[0][2] + D[1][2] + D[2][2]);
        } else {
            uint k = lv - 1u;
            d0 = D[k][0];
            d1 = D[k][1];
            d2 = D[k][2];
        }

        // f = P * d  (d = (d0,d1,d2))
        double w = vol[tet];
        g.x += (P[0][0] * d0 + P[0][1] * d1 + P[0][2] * d2) * w;
        g.y += (P[1][0] * d0 + P[1][1] * d1 + P[1][2] * d2) * w;
        g.z += (P[2][0] * d0 + P[2][1] * d1 + P[2][2] * d2) * w;
    }

    store_double3(grad, v, g);
}
