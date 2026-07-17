//
// gipc-pfpx.h 鈥?shared (C++ + HLSL) GIPC PFPx contact vector.
//
// For the GIPC NEWF construction the gradient and (rank-1) Hessian of a contact
// pair both reduce to a single vector  v = PFPx岬€路q0  and the scalar I5:
//     grad_local = kappa 路 gradCoeff(I5) 路 sqrt(I5) 路 v
//     H_local    = kappa 路 clampedLambda0(I5) 路 v路v岬€      (rank-1)
// Since q0 = e_last in the NEWF basis, v is exactly the last (only non-zero)
// column of the autogen 鈭倂ec(F)/鈭倄 Jacobian. These functions transcribe that
// column verbatim from GIPC_PDerivative.h (pFpx_{pp,pe,pt,ee}2) 鈥?the only
// transcendental is shSqrt 鈥?and compute I5 = d虏/d虃虏 from the matching distance.
//
// Vertex/DOF order (matches fem::ipc unified ConstraintPair indices):
//   PP: [p0,p1]            v[0..5]
//   PE: [p, e0, e1]        v[0..8]
//   PT: [p, t0, t1, t2]    v[0..11]
//   EE: [ea0,ea1,eb0,eb1]  v[0..11]
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_PFPX_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_PFPX_H_

#include <Contact/shared/ipc-distance.h>

SH_NS_BEGIN

struct ShGipcPfpx {
    real v[12];   // contact vector (n*3 used)
    real I5;      // (d/dHat)^2
    int     n;       // number of vertices (2/3/4)
    bool    valid;   // false if degenerate or I5 >= 1 (inactive)
};

SH_INLINE ShGipcPfpx shGipcMakePfpx(int n) {
    ShGipcPfpx r;
    r.n = n; r.I5 = 1.0; r.valid = false;
    for (int i = 0; i < 12; ++i) r.v[i] = 0.0;
    return r;
}

// ---- Point-Point: x = [p0, p1] ---------------------------------------------
SH_INLINE ShGipcPfpx shGipcPFPx_PP(real3 p0, real3 p1, real dHat) {
    ShGipcPfpx r = shGipcMakePfpx(2);
    real dHatSqr = dHat * dHat;
    real dSqr = shDistanceSqrPointPoint(p0, p1);
    r.I5 = dSqr / dHatSqr;
    real dis = shSqrt(dSqr);
    if (dis < 1e-30) return r;

    real t8  = 1.0 / dHat;
    real inv = 1.0 / dis;
    real v0 = t8 * (p0.x - p1.x) * inv;
    real v1 = t8 * (p0.y - p1.y) * inv;
    real v2 = t8 * (p0.z - p1.z) * inv;
    r.v[0] = v0; r.v[1] = v1; r.v[2] = v2;
    r.v[3] = -v0; r.v[4] = -v1; r.v[5] = -v2;
    r.valid = true;
    return r;
}

// ---- Point-Edge: x = [p, e0, e1] -------------------------------------------
SH_INLINE ShGipcPfpx shGipcPFPx_PE(real3 p, real3 e0, real3 e1, real dHat) {
    ShGipcPfpx r = shGipcMakePfpx(3);
    real dHatSqr = dHat * dHat;
    real3 ed = e1 - e0;
    if (shDot(ed, ed) < 1e-60) return r;
    r.I5 = shDistanceSqrPointLine(p, e0, e1) / dHatSqr;

    real t8  = 1.0 / dHat;
    real t18 = p.x - e0.x, t19 = p.y - e0.y, t20 = p.z - e0.z;
    real t21 = p.x - e1.x, t22 = p.y - e1.y, t23 = p.z - e1.z;
    real t24 = e0.x - e1.x, t25 = e0.y - e1.y, t26 = e0.z - e1.z;
    real t43 = 1.0 / ((t24 * t24 + t25 * t25) + t26 * t26);
    real t45 = t18 * t22 - t19 * t21;
    real t46 = t18 * t23 - t20 * t21;
    real t47 = t19 * t23 - t20 * t22;
    real t43sq = t43 * t43;
    real nrm = (t45 * t45 + t46 * t46) + t47 * t47;
    real t54 = (2.0 * e0.x - 2.0 * e1.x) * t43sq * nrm;
    real t55 = (2.0 * e0.y - 2.0 * e1.y) * t43sq * nrm;
    real t56 = (2.0 * e0.z - 2.0 * e1.z) * t43sq * nrm;
    real inv = 1.0 / shSqrt(t43 * nrm);
    real cA = t8 * t43 * inv;
    r.v[0] = cA * (t25 * t45 * 2.0 + t26 * t46 * 2.0) / 2.0;
    r.v[1] = cA * (t24 * t45 * 2.0 - t26 * t47 * 2.0) * -0.5;
    r.v[2] = cA * (t24 * t46 * 2.0 + t25 * t47 * 2.0) * -0.5;
    real cB = t8 * inv;
    r.v[3] = cB * (t54 + t43 * (t22 * t45 * 2.0 + t23 * t46 * 2.0)) * -0.5;
    r.v[4] = cB * (t55 - t43 * (t21 * t45 * 2.0 - t23 * t47 * 2.0)) * -0.5;
    r.v[5] = cB * (t56 - t43 * (t21 * t46 * 2.0 + t22 * t47 * 2.0)) * -0.5;
    r.v[6] = cB * (t54 + t43 * (t19 * t45 * 2.0 + t20 * t46 * 2.0)) / 2.0;
    r.v[7] = cB * (t55 - t43 * (t18 * t45 * 2.0 - t20 * t47 * 2.0)) / 2.0;
    r.v[8] = cB * (t56 - t43 * (t18 * t46 * 2.0 + t19 * t47 * 2.0)) / 2.0;
    r.valid = true;
    return r;
}

// ---- Point-Triangle: x = [p, t0, t1, t2] -----------------------------------
SH_INLINE ShGipcPfpx shGipcPFPx_PT(real3 p, real3 t0, real3 t1, real3 t2,
                                   real dHat) {
    ShGipcPfpx r = shGipcMakePfpx(4);
    real dHatSqr = dHat * dHat;
    real3 nrmTri = shCross(t1 - t0, t2 - t0);
    if (shDot(nrmTri, nrmTri) < 1e-60) return r;
    r.I5 = shDistanceSqrPointPlane(p, t0, t1, t2) / dHatSqr;

    real id = 1.0 / dHat;
    real t12 = p.x - t0.x, t13 = p.y - t0.y, t14 = p.z - t0.z;
    real t15 = t0.x - t1.x, t16 = t0.y - t1.y, t17 = t0.z - t1.z;
    real t18 = t0.x - t2.x, t19 = t0.y - t2.y, t20 = t0.z - t2.z;
    real t21 = t1.x - t2.x, t22 = t1.y - t2.y, t23 = t1.z - t2.z;
    real t33 = t15 * t19 - t16 * t18;
    real t34 = t15 * t20 - t17 * t18;
    real t35 = t16 * t20 - t17 * t19;
    real t44 = 1.0 / ((t33 * t33 + t34 * t34) + t35 * t35);
    real t46 = (t14 * t33 + t12 * t35) - t13 * t34;
    real t47 = t46 * t46;
    real t49 = 1.0 / shSqrt(t44 * t47);
    r.v[0] = id * t35 * t44 * t46 * t49;
    r.v[1] = -id * t34 * t44 * t46 * t49;
    r.v[2] = id * t33 * t44 * t46 * t49;
    real b_d = t44 * t44 * t47;
    real t44m = t44 * t46;
    real cC = id * t49;
    r.v[3] = cC * (b_d * (t22 * t33 * 2.0 + t23 * t34 * 2.0)
                   + t44m * ((t35 + t13 * t23) - t14 * t22) * 2.0) * -0.5;
    r.v[4] = cC * (b_d * (t21 * t33 * 2.0 - t23 * t35 * 2.0)
                   + t44m * ((t34 + t12 * t23) - t14 * t21) * 2.0) / 2.0;
    r.v[5] = cC * (b_d * (t21 * t34 * 2.0 + t22 * t35 * 2.0)
                   - t44m * ((t33 + t12 * t22) - t13 * t21) * 2.0) / 2.0;
    r.v[6] = cC * (t44m * (t13 * t20 - t14 * t19) * 2.0
                   + b_d * (t19 * t33 * 2.0 + t20 * t34 * 2.0)) / 2.0;
    r.v[7] = cC * (t44m * (t12 * t20 - t14 * t18) * 2.0
                   + b_d * (t18 * t33 * 2.0 - t20 * t35 * 2.0)) * -0.5;
    r.v[8] = cC * (t44m * (t12 * t19 - t13 * t18) * 2.0
                   - b_d * (t18 * t34 * 2.0 + t19 * t35 * 2.0)) / 2.0;
    r.v[9] = cC * (t44m * (t13 * t17 - t14 * t16) * 2.0
                   + b_d * (t16 * t33 * 2.0 + t17 * t34 * 2.0)) * -0.5;
    r.v[10] = cC * (t44m * (t12 * t17 - t14 * t15) * 2.0
                    + b_d * (t15 * t33 * 2.0 - t17 * t35 * 2.0)) / 2.0;
    r.v[11] = cC * (t44m * (t12 * t16 - t13 * t15) * 2.0
                    - b_d * (t15 * t34 * 2.0 + t16 * t35 * 2.0)) * -0.5;
    r.valid = true;
    return r;
}

// ---- Edge-Edge: x = [ea0, ea1, eb0, eb1] -----------------------------------
SH_INLINE ShGipcPfpx shGipcPFPx_EE(real3 ea0, real3 ea1, real3 eb0, real3 eb1,
                                   real dHat) {
    ShGipcPfpx r = shGipcMakePfpx(4);
    real dHatSqr = dHat * dHat;
    real3 cr = shCross(ea1 - ea0, eb1 - eb0);
    if (shDot(cr, cr) < 1e-60) return r;  // near-parallel -> mollifier (unhandled)
    r.I5 = shDistanceSqrLineLine(ea0, ea1, eb0, eb1) / dHatSqr;

    real t12 = ea0.x - ea1.x, t13 = ea0.y - ea1.y, t14 = ea0.z - ea1.z;
    real t15 = ea0.x - eb0.x, t16 = ea0.y - eb0.y, t17 = ea0.z - eb0.z;
    real t18 = eb0.x - eb1.x, t19 = eb0.y - eb1.y, t20 = eb0.z - eb1.z;
    real t33 = t15 * t19, t34 = t16 * t18, t35 = t15 * t20;
    real t36 = t17 * t18, t37 = t16 * t20, t38 = t17 * t19;
    real t45 = t12 * t19 - t13 * t18;
    real t46 = t12 * t20 - t14 * t18;
    real t47 = t13 * t20 - t14 * t19;
    real t76 = 1.0 / ((t45 * t45 + t46 * t46) + t47 * t47);
    real t78 = (t17 * t45 + t15 * t47) - t16 * t46;
    real t76sq = t76 * t76;
    real t79 = t78 * t78;
    real t82 = (t13 * t45 * 2.0 + t14 * t46 * 2.0) * t76sq * t79;
    real t83 = (t12 * t46 * 2.0 + t13 * t47 * 2.0) * t76sq * t79;
    real t84 = (t19 * t45 * 2.0 + t20 * t46 * 2.0) * t76sq * t79;
    real t19A = (t18 * t46 * 2.0 + t19 * t47 * 2.0) * t76sq * t79;
    real t86 = (t12 * t45 * 2.0 - t14 * t47 * 2.0) * t76sq * t79;
    real t20A = (t18 * t45 * 2.0 - t20 * t47 * 2.0) * t76sq * t79;
    real t78m = t78 * t76;
    real cC = (1.0 / dHat) * (1.0 / shSqrt(t76 * t79));
    r.v[0] = cC * (t84 - t78m * ((-t37 + t38) + t47) * 2.0) * -0.5;
    r.v[1] = cC * (t20A - t78m * ((-t35 + t36) + t46) * 2.0) / 2.0;
    r.v[2] = cC * (t19A + t78m * ((-t33 + t34) + t45) * 2.0) / 2.0;
    r.v[3] = cC * (t84 + t78m * (t37 - t38) * 2.0) / 2.0;
    r.v[4] = cC * (t20A + t78m * (t35 - t36) * 2.0) * -0.5;
    r.v[5] = cC * (t19A - t78m * (t33 - t34) * 2.0) * -0.5;
    real t18B = t13 * t17 - t14 * t16;
    r.v[6] = cC * (t82 - t78m * (t18B + t47) * 2.0) / 2.0;
    real t20B = t12 * t17 - t14 * t15;
    r.v[7] = cC * (t86 - t78m * (t20B + t46) * 2.0) * -0.5;
    real t19B = t12 * t16 - t13 * t15;
    r.v[8] = cC * (t83 + t78m * (t19B + t45) * 2.0) * -0.5;
    r.v[9] = cC * (t82 - t78m * t18B * 2.0) * -0.5;
    r.v[10] = cC * (t86 - t78m * t20B * 2.0) / 2.0;
    r.v[11] = cC * (t83 + t78m * t19B * 2.0) / 2.0;
    r.valid = true;
    return r;
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_GIPC_PFPX_H_
