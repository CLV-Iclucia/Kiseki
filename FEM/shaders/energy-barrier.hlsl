// ============================================================================
// energy-barrier.hlsl — per-constraint GIPC barrier energy for the GPU line
// search. One thread per active ConstraintPair (grouped PP|PE|PT|EE by the
// compact typeOffsets produced by GpuActivation).
//
// Mirrors CPU fem::ipc::constraintPairBarrierEnergy EXACTLY (single source of
// truth in <fem/shared/...>):
//   * squared distance by kind: PP=point-point, PE=point-line,
//     PT=point-triangle (decide+dispatch), EE=line-line,
//   * near-parallel edge-edge routes to the mollified barrier energy
//     (shMollifiedEnergyEE), restoring C1 continuity,
//   * otherwise  E = kappa * shBarrierEnergy(dSqr, dHatSqr)  (0 when dSqr>=dHatSqr).
//
// Output is per-constraint; a downstream Sum reduction yields the total barrier
// energy, which the backend folds into the incremental-potential line search.
// ============================================================================
#include <fem/shared/ipc-distance.h>
#include <fem/shared/gipc-barrier.h>
#include <fem/shared/gipc-mollifier.h>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> energyOut : register(u0); // [total]
[[vk::binding(1, 0)]] StructuredBuffer<int>      pairs     : register(t0); // [total*4]
[[vk::binding(2, 0)]] StructuredBuffer<double>   x         : register(t1); // [nVerts*3]
[[vk::binding(3, 0)]] StructuredBuffer<double>   xRest     : register(t2); // [nVerts*3]
[[vk::binding(4, 0)]] StructuredBuffer<uint>     typeOff   : register(t3); // [5]
[[vk::binding(5, 0)]] StructuredBuffer<double>   dParams   : register(t4); // [kappa, dHat]

struct PC { uint total; };
[[vk::push_constant]] PC pc;

double3 loadP(int i) { return load_double3(x, uint(i)); }
double3 loadR(int i) { return load_double3(xRest, uint(i)); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.total) return;

    // ---- locate kind bucket from typeOffsets {0, ppEnd, peEnd, ptEnd, total} ----
    uint to1 = typeOff[1], to2 = typeOff[2], to3 = typeOff[3];
    int k;
    if      (s < to1) k = 0;   // PP
    else if (s < to2) k = 1;   // PE
    else if (s < to3) k = 2;   // PT
    else              k = 3;   // EE

    int i0 = pairs[s * 4 + 0], i1 = pairs[s * 4 + 1];
    int i2 = pairs[s * 4 + 2], i3 = pairs[s * 4 + 3];

    double kappa = dParams[0], dHat = dParams[1], dHatSqr = dHat * dHat;

    double3 p0 = loadP(i0);
    double3 p1 = loadP(i1);

    double e = 0.0;
    if (k == 0) {
        double dSqr = shDistanceSqrPointPoint(p0, p1);
        e = (dSqr >= dHatSqr) ? 0.0 : kappa * shBarrierEnergy(dSqr, dHatSqr);
    } else if (k == 1) {
        double3 p2 = loadP(i2);
        double dSqr = shDistanceSqrPointLine(p0, p1, p2);
        e = (dSqr >= dHatSqr) ? 0.0 : kappa * shBarrierEnergy(dSqr, dHatSqr);
    } else if (k == 2) {
        double3 p2 = loadP(i2), p3 = loadP(i3);
        double dSqr = shDistanceSqrPointTriangle(p0, p1, p2, p3);
        e = (dSqr >= dHatSqr) ? 0.0 : kappa * shBarrierEnergy(dSqr, dHatSqr);
    } else {
        double3 p2 = loadP(i2), p3 = loadP(i3);
        double3 r0 = loadR(i0), r1 = loadR(i1), r2 = loadR(i2), r3 = loadR(i3);
        if (shEEUsesMollifier(p0, p1, p2, p3, r0, r1, r2, r3)) {
            e = shMollifiedEnergyEE(p0, p1, p2, p3, r0, r1, r2, r3, dHat, kappa);
        } else {
            double dSqr = shDistanceSqrLineLine(p0, p1, p2, p3);
            e = (dSqr >= dHatSqr) ? 0.0 : kappa * shBarrierEnergy(dSqr, dHatSqr);
        }
    }
    energyOut[s] = e;
}
