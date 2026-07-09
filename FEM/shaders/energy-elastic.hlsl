// ============================================================================
// energy-elastic.hlsl — Stable Neo-Hookean elastic energy density (double).
// One thread per tet: reconstruct F = Ds*DmInv, then the SNH energy density
//   Psi = 0.5*mu*(Ic - 3) - mu*(J - 1) + 0.5*lambda*(J - 1)^2
// (Ic = ||F||_F^2, J = det F), scaled by the tet rest volume. This is exactly
// the energy whose PK1 is mu*F + (lambda*(J-1) - mu)*cof(F) — i.e. consistent
// with elastic-gradient.hlsl / elastic-hessian.hlsl — and matches CPU
// StableNeoHookean::computeEnergyDensity * tetRefVolume.
//
// Output is per-tet; a downstream Sum reduction yields the total elastic energy.
// Matrices (DmInv) are column-major: M(r,c) = DmInv[tet*9 + c*3 + r].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> energyOut : register(u0); // [M]
[[vk::binding(1, 0)]] StructuredBuffer<double>   x         : register(t0); // [N*3]
[[vk::binding(2, 0)]] StructuredBuffer<uint>     tetConn   : register(t1); // [M*4]
[[vk::binding(3, 0)]] StructuredBuffer<double>   DmInv     : register(t2); // [M*9]
[[vk::binding(4, 0)]] StructuredBuffer<double>   vol       : register(t3); // [M]
[[vk::binding(5, 0)]] StructuredBuffer<double>   material  : register(t4); // [mu, lambda]

struct PC { uint numTets; };
[[vk::push_constant]] PC pc;

double3 ld(uint i) { return load_double3(x, i); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= pc.numTets) return;

    double mu = material[0], lambda = material[1];
    uint cb = t * 4u;
    double3x3 D = load_double3x3_col_major(DmInv, t);

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

    double Ic = 0.0;
    [unroll] for (uint r = 0; r < 3; ++r)
        [unroll] for (uint c2 = 0; c2 < 3; ++c2)
            Ic += F[r][c2] * F[r][c2];

    double J = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
             - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
             + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

    double Jm1 = J - 1.0;
    double psi = 0.5 * mu * (Ic - 3.0) - mu * Jm1 + 0.5 * lambda * Jm1 * Jm1;
    energyOut[t] = psi * vol[t];
}
