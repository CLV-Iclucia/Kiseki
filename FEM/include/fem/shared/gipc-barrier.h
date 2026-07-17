//
// gipc-barrier.h — shared (C++ + HLSL) GIPC (RANK=2) barrier scalar coefficients.
//
// Single source of truth for the scalar barrier functions behind
// fem::ipc::gipc::Barrier (energy / gradCoeff / lambda0 / clampedLambda0).
// `dHatSqr` = d̂² (Barrier::dHatSqr()), `I5` = d²/d̂² in (0,1) when active.
// Matches GIPC.cu; sHat2 = d̂⁴.
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_BARRIER_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_BARRIER_H_

#include <fem/shared/cross-lang.h>

#define SH_GAUSS_THRESHOLD 1e-4

SH_NS_BEGIN

// barrier energy b(d²) without kappa: ŝ²-free form t²·ln²(I5), t = d²-d̂².
SH_INLINE real shBarrierEnergy(real dSqr, real dHatSqr) {
    if (dSqr >= dHatSqr) return 0.0;
    if (dSqr <= 0.0)     return 1e18;
    real I5 = dSqr / dHatSqr;
    real L  = shLog(I5);
    real t  = dSqr - dHatSqr;
    return t * t * L * L;
}

// gradient scalar coefficient = 2·∂b/∂I5 (includes the chain factor 2).
SH_INLINE real shBarrierGradCoeff(real I5, real dHatSqr) {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    real sHat2 = dHatSqr * dHatSqr;
    real L = shLog(I5);
    return 4.0 * sHat2 * L * (I5 - 1.0) * (I5 + I5 * L - 1.0) / I5;
}

// λ₀(I5): closed-form non-zero eigenvalue of the inner Hessian (no kappa).
SH_INLINE real shBarrierLambda0(real I5, real dHatSqr) {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    real sHat2 = dHatSqr * dHatSqr;
    real L = shLog(I5);
    return -(4.0 * sHat2
             * (4.0 * I5 + L - 3.0 * I5 * I5 * L * L + 6.0 * I5 * L
                - 2.0 * I5 * I5 + I5 * L * L - 7.0 * I5 * I5 * L - 2.0))
           / I5;
}

// Gauss-threshold-guarded, SPD-projected λ₀ (no kappa).
SH_INLINE real shBarrierClampedLambda0(real I5, real dHatSqr) {
    if (I5 >= 1.0) return 0.0;
    real lam = (I5 < SH_GAUSS_THRESHOLD)
                      ? shBarrierLambda0(SH_GAUSS_THRESHOLD, dHatSqr)
                      : shBarrierLambda0(I5, dHatSqr);
    return shMax(lam, 0.0);
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_BARRIER_H_
