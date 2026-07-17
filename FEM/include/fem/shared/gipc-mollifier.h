//
// gipc-mollifier.h — shared (C++ + HLSL) GIPC edge-edge mollifier.
//
// When two edges become near-parallel the plain edge-edge barrier gradient /
// Hessian are discontinuous; GIPC multiplies them by a smooth mollifier
// m_eps(I1) (I1 = ||ea x eb||^2) that ramps to 0 as the edges align, restoring
// C1 continuity. This is REQUIRED for correctness of edge-edge contact.
//
// This is the single source of truth shared by the CPU unified barrier path
// (fem::ipc::constraintPairBarrier*) and the GPU barrier assembler, mirroring
// fem::ipc::gipc::computeMollifiedBarrier exactly.
//
// Key simplification (verified against the autogen pFpx_pee): the mollifier
// PFPx (9x12) has only TWO non-zero rows — row 4 (the tangential / I1 column,
// "r4") and row 8 (the distance / I2 column, "r8"). r8 is bit-identical to the
// plain edge-edge contact vector shGipcPFPx_EE().v (same d/d̂ column), so it is
// reused; r4 is transcribed here. Because rows 3 and 5 are zero, the twist
// eigenpairs (lambda11/lambda12) vanish after the PFPx^T(.)PFPx sandwich, leaving
// only the 2x2 {e4,e8} block. Hence:
//     grad = p1*c*r4 + p2*F33*r8
//     H    = sum_k lam_k * w_k w_k^T,   w_k = a_k*r4 + b_k*r8   (k = 0..rank-1)
// where (lam_k, (a_k,b_k)) are the SPD-projected eigenpairs of
//     [[lambda10, lambdaG1G],[lambdaG1G, lambda20]].
//
// Vertex/DOF order: x = [ea0, ea1, eb0, eb1]; grad[0..11], w[*][0..11].
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_MOLLIFIER_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_MOLLIFIER_H_

#include <fem/shared/gipc-pfpx.h>     // shGipcPFPx_EE (r8) + shDistanceSqrLineLine
#include <fem/shared/gipc-barrier.h>  // shared barrier conventions

SH_NS_BEGIN

// ε_x = 1e-3 · |ea_rest|² · |eb_rest|²  (rest-pose mollifier threshold).
SH_INLINE real shMollifierEpsX(real3 rea0, real3 rea1,
                                  real3 reb0, real3 reb1) {
    real3 ra = rea1 - rea0;
    real3 rb = reb1 - reb0;
    return 1e-3 * shDot(ra, ra) * shDot(rb, rb);
}

// m_eps(c) = 2c/ε − c²/ε²  (c < ε), else 1.   c = I1 = ||ea×eb||².
SH_INLINE real shMollifierValue(real c, real epsX) {
    if (c >= epsX) return 1.0;
    return (2.0 / epsX) * c - (1.0 / (epsX * epsX)) * c * c;
}

// True when the edge pair is near-parallel by the mollifier criterion
// (current cross² < rest-pose ε_x). Caller routes such EE pairs through the
// mollified branch; all others use the plain shGipcPFPx_EE rank-1 path.
SH_INLINE bool shEEUsesMollifier(real3 ea0, real3 ea1,
                                 real3 eb0, real3 eb1,
                                 real3 rea0, real3 rea1,
                                 real3 reb0, real3 reb1) {
    real3 cr = shCross(ea1 - ea0, eb1 - eb0);
    real I1 = shDot(cr, cr);
    return I1 < shMollifierEpsX(rea0, rea1, reb0, reb1);
}

// ---- mollifier scalar coefficients (no kappa; transcribed from Barrier) ----

SH_INLINE real shMollifierP1(real I1, real I2, real epsX, real sHat2) {
    if (I2 >= 1.0) return 0.0;
    real L2 = shLog(I2);
    return -4.0 * sHat2 * L2 * L2 * (I1 - epsX) * (I2 - 1.0) * (I2 - 1.0)
           / (epsX * epsX);
}

SH_INLINE real shMollifierP2(real I1, real I2, real epsX, real sHat2) {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    real L2 = shLog(I2);
    return -4.0 * I1 * sHat2 * L2 * (I1 - 2.0 * epsX)
           * (I2 - 1.0) * (I2 + I2 * L2 - 1.0) / (I2 * epsX * epsX);
}

SH_INLINE real shMollifierLambda10(real I1, real I2, real epsX, real sHat2) {
    if (I2 >= 1.0) return 0.0;
    real L2 = shLog(I2);
    return -(4.0 * sHat2 * L2 * L2 * (I2 - 1.0) * (I2 - 1.0) * (3.0 * I1 - epsX))
           / (epsX * epsX);
}

SH_INLINE real shMollifierLambda20(real I1, real I2, real epsX, real sHat2) {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    real L2 = shLog(I2);
    return (4.0 * sHat2 * I1 * (I1 - 2.0 * epsX)
            * (4.0 * I2 + L2 - 3.0 * I2 * I2 * L2 * L2 + 6.0 * I2 * L2
               - 2.0 * I2 * I2 + I2 * L2 * L2 - 7.0 * I2 * I2 * L2 - 2.0))
           / (I2 * epsX * epsX);
}

SH_INLINE real shMollifierLambdaG1G(real I1, real I2, real epsX, real sHat2) {
    if (I2 >= 1.0 || I2 <= 0.0 || I1 <= 0.0) return 0.0;
    real L2 = shLog(I2);
    real c   = shSqrt(I1);
    real F33 = shSqrt(I2);
    return -4.0 * c * F33 * sHat2 * L2 * (I1 - epsX)
           * (I2 - 1.0) * (I2 + I2 * L2 - 1.0) / (I2 * epsX * epsX);
}

// ---- 2x2 symmetric SPD projection: [[a,b],[b,d]] -> positive eigenpairs ----
struct ShPD2 {
    real val[2];
    real vec[2][2];   // column k: (vec[k][0], vec[k][1])
    int     num;
};

SH_INLINE ShPD2 shMakePD2x2(real a, real b, real d) {
    ShPD2 r;
    r.num = 0;
    r.val[0] = 0.0; r.val[1] = 0.0;
    r.vec[0][0] = 1.0; r.vec[0][1] = 0.0;
    r.vec[1][0] = 1.0; r.vec[1][1] = 0.0;

    real trace = a + d;
    real det   = a * d - b * b;
    real disc  = trace * trace - 4.0 * det;
    disc = shMax(disc, 0.0);
    real sd = shSqrt(disc);
    real lam[2];
    lam[0] = 0.5 * (trace + sd);
    lam[1] = 0.5 * (trace - sd);

    for (int k = 0; k < 2; ++k) {
        if (lam[k] <= 0.0) continue;
        real r0 = a - lam[k];
        real r1 = b;
        real mag = (r0 < 0.0 ? -r0 : r0) + (r1 < 0.0 ? -r1 : r1);
        if (mag < 1e-30) { r0 = -b; r1 = d - lam[k]; }
        // eigenvector = normalize((-r1, r0))
        real v0 = -r1, v1 = r0;
        real nrm = shSqrt(v0 * v0 + v1 * v1);
        if (nrm > 1e-30) { v0 /= nrm; v1 /= nrm; }
        else             { v0 = 1.0; v1 = 0.0; }
        r.val[r.num] = lam[k];
        r.vec[r.num][0] = v0;
        r.vec[r.num][1] = v1;
        r.num++;
    }
    return r;
}

// ---- mollified edge-edge gradient + Hessian (rank <= 2) --------------------
struct ShMollifiedEE {
    real grad[12];     // ∂E/∂x, x = [ea0,ea1,eb0,eb1]
    real w[2][12];     // rank-1 Hessian generators: H = Σ lam[k]·w[k]·w[k]ᵀ
    real lam[2];       // eigenvalues (>= 0), already including kappa
    int     rank;         // number of active Hessian terms (0..2)
    bool    active;       // false -> contributes nothing (skip)
};

SH_INLINE ShMollifiedEE shMollifiedBarrierEE(
    real3 ea0, real3 ea1, real3 eb0, real3 eb1,
    real3 rea0, real3 rea1, real3 reb0, real3 reb1,
    real dHat, real kappa) {
    ShMollifiedEE res;
    res.rank = 0; res.active = false;
    res.lam[0] = 0.0; res.lam[1] = 0.0;
    for (int i = 0; i < 12; ++i) { res.grad[i] = 0.0; res.w[0][i] = 0.0; res.w[1][i] = 0.0; }

    real dHatSqr = dHat * dHat;
    real sHat2   = dHatSqr * dHatSqr;

    // I1 = ||ea x eb||² (collinearity), eps_x from rest pose.
    real3 cr = shCross(ea1 - ea0, eb1 - eb0);
    real I1 = shDot(cr, cr);
    if (I1 <= 0.0) return res;                 // exactly parallel / singular
    real epsX = shMollifierEpsX(rea0, rea1, reb0, reb1);

    // r8 = the distance column = plain EE contact vector (same d/d̂ column).
    ShGipcPfpx ee = shGipcPFPx_EE(ea0, ea1, eb0, eb1, dHat);
    if (!ee.valid) return res;                 // I5>=1 or degenerate -> inactive
    real I2 = ee.I5;                         // = d²/d̂²
    if (I2 >= 1.0 || I2 <= 0.0) return res;

    // r4 = the tangential (I1) column (transcribed from pFpx_pee column 4).
    real t12 = ea0.x - ea1.x, t13 = ea0.y - ea1.y, t14 = ea0.z - ea1.z;
    real t18 = eb0.x - eb1.x, t19 = eb0.y - eb1.y, t20 = eb0.z - eb1.z;
    real t45 = t12 * t19 - t13 * t18;
    real t46 = t12 * t20 - t14 * t18;
    real t47 = t13 * t20 - t14 * t19;
    real invSqrtI1 = 1.0 / shSqrt(I1);
    real t81 = (t13 * t45 + t14 * t46) * invSqrtI1;
    real t82 = (t12 * t46 + t13 * t47) * invSqrtI1;
    real t83 = (t19 * t45 + t20 * t46) * invSqrtI1;
    real t84 = (t18 * t46 + t19 * t47) * invSqrtI1;
    real t85 = (t12 * t45 - t14 * t47) * invSqrtI1;
    real t86 = (t18 * t45 - t20 * t47) * invSqrtI1;
    real r4[12];
    r4[0] = t83;  r4[1] = -t86; r4[2] = -t84;
    r4[3] = -t83; r4[4] = t86;  r4[5] = t84;
    r4[6] = -t81; r4[7] = t85;  r4[8] = t82;
    r4[9] = t81;  r4[10] = -t85; r4[11] = -t82;

    // Scalar coefficients (fold kappa in).
    real c   = shSqrt(I1);
    real F33 = shSqrt(I2);
    real p1 = kappa * shMollifierP1(I1, I2, epsX, sHat2);
    real p2 = kappa * shMollifierP2(I1, I2, epsX, sHat2);

    // Gradient: p1·c·r4 + p2·F33·r8.
    real gc4 = p1 * c;
    real gc8 = p2 * F33;
    for (int i = 0; i < 12; ++i)
        res.grad[i] = gc4 * r4[i] + gc8 * ee.v[i];

    // Hessian: SPD-project the {e4,e8} 2x2 and expand to rank-1 generators.
    real l10 = kappa * shMollifierLambda10(I1, I2, epsX, sHat2);
    real l20 = kappa * shMollifierLambda20(I1, I2, epsX, sHat2);
    real lg  = kappa * shMollifierLambdaG1G(I1, I2, epsX, sHat2);
    ShPD2 pd = shMakePD2x2(l10, lg, l20);
    res.rank = pd.num;
    for (int k = 0; k < pd.num; ++k) {
        real a = pd.vec[k][0], b = pd.vec[k][1];
        res.lam[k] = pd.val[k];
        for (int i = 0; i < 12; ++i)
            res.w[k][i] = a * r4[i] + b * ee.v[i];
    }

    res.active = true;
    return res;
}

// Mollified edge-edge barrier energy (for line search / energy parity).
//   E = kappa · m(I1) · ŝ² · (1-I2)² · ln²(I2)
SH_INLINE real shMollifiedEnergyEE(
    real3 ea0, real3 ea1, real3 eb0, real3 eb1,
    real3 rea0, real3 rea1, real3 reb0, real3 reb1,
    real dHat, real kappa) {
    real dHatSqr = dHat * dHat;
    real3 cr = shCross(ea1 - ea0, eb1 - eb0);
    real I1 = shDot(cr, cr);
    if (I1 <= 0.0) return 0.0;
    real dSqr = shDistanceSqrLineLine(ea0, ea1, eb0, eb1);
    real I2 = dSqr / dHatSqr;
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    real epsX = shMollifierEpsX(rea0, rea1, reb0, reb1);
    real m  = shMollifierValue(I1, epsX);
    real L2 = shLog(I2);
    real t2 = 1.0 - I2;
    real sHat2 = dHatSqr * dHatSqr;
    return kappa * m * sHat2 * t2 * t2 * L2 * L2;
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_MOLLIFIER_H_
