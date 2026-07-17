//
// ipc-distance-type.h — shared (C++ + HLSL) IPC distance-type classification.
//
// Single source of truth for the point-triangle / edge-edge distance-type
// decision logic. The CPU side (fem/ipc/distances.h) wraps these into its
// scoped enums; GPU activation kernels #include this header directly so the
// classification is provably identical on both sides.
//
// Include path:
//   C++  : #include <fem/shared/ipc-distance-type.h>   (FEM/include on path)
//   HLSL : #include <fem/shared/ipc-distance-type.h>   (add FEM/include via -I)
//
// The integer codes below MUST stay in lockstep with the enum orderings in
// fem/ipc/distances.h (EdgeEdgeDistanceType / PointTriangleDistanceType), so a
// plain static_cast across the boundary is exact.
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_TYPE_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_TYPE_H_

#include <fem/shared/cross-lang.h>

// ---- EdgeEdgeDistanceType codes (match the enum class ordering) -----------
#define SH_EE_A_C     0
#define SH_EE_A_D     1
#define SH_EE_B_C     2
#define SH_EE_B_D     3
#define SH_EE_AB_C    4
#define SH_EE_AB_D    5
#define SH_EE_A_CD    6
#define SH_EE_B_CD    7
#define SH_EE_AB_CD   8
#define SH_EE_UNKNOWN 9

// ---- PointTriangleDistanceType codes (match the enum class ordering) ------
#define SH_PT_P_A     0
#define SH_PT_P_B     1
#define SH_PT_P_C     2
#define SH_PT_P_AB    3
#define SH_PT_P_BC    4
#define SH_PT_P_CA    5
#define SH_PT_P_ABC   6
#define SH_PT_UNKNOWN 7

SH_NS_BEGIN

// ===========================================================================
// decideEdgeEdgeParallelDistanceType
// ===========================================================================
SH_INLINE int shDecideEdgeEdgeParallelDistanceType(
    real3 ea0, real3 ea1, real3 eb0, real3 eb1) {
    real3 ea    = ea1 - ea0;
    real  eaSq  = shDot(ea, ea);
    real  alpha = shDot(eb0 - ea0, ea) / eaSq;
    real  beta  = shDot(eb1 - ea0, ea) / eaSq;

    int eac;  // 0: EA0, 1: EA1, 2: EA
    int ebc;  // 0: EB0, 1: EB1, 2: EB
    if (alpha < 0.0) {
        eac = (0.0 <= beta && beta <= 1.0) ? 2 : 0;
        ebc = (beta <= alpha) ? 0 : (beta <= 1.0 ? 1 : 2);
    } else if (alpha > 1.0) {
        eac = (0.0 <= beta && beta <= 1.0) ? 2 : 1;
        ebc = (beta >= alpha) ? 0 : (0.0 <= beta ? 1 : 2);
    } else {
        eac = 2;
        ebc = 0;
    }

    SH_ASSERT(eac != 2 || ebc != 2);
    return (ebc < 2) ? ((eac << 1) | ebc) : (6 + eac);
}

// ===========================================================================
// decideEdgeEdgeDistanceType
// ===========================================================================
SH_INLINE int shDecideEdgeEdgeDistanceType(
    real3 ea0, real3 ea1, real3 eb0, real3 eb1) {
    const real PARALLEL_THRESHOLD = 1.0e-20;

    real3 u = ea1 - ea0;
    real3 v = eb1 - eb0;
    real3 w = ea0 - eb0;

    real a = shDot(u, u);
    real b = shDot(u, v);
    real c = shDot(v, v);
    real d = shDot(u, w);
    real e = shDot(v, w);
    real D = a * c - b * b;

    if (a == 0.0 && c == 0.0) return SH_EE_A_C;
    else if (a == 0.0)        return SH_EE_A_CD;
    else if (c == 0.0)        return SH_EE_AB_C;

    real  parallel_tolerance = PARALLEL_THRESHOLD * shMax(1.0, a * c);
    real3 uxv = shCross(u, v);
    if (shDot(uxv, uxv) < parallel_tolerance)
        return shDecideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);

    int     default_case = SH_EE_AB_CD;
    real sN = (b * e - c * d);
    real tN, tD;
    if (sN <= 0.0) {
        tN = e;
        tD = c;
        default_case = SH_EE_A_CD;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        default_case = SH_EE_B_CD;
    } else {
        tN = (a * e - b * d);
        tD = D;
        if (tN > 0.0 && tN < tD && shDot(uxv, uxv) < parallel_tolerance) {
            if (sN < D / 2.0) {
                tN = e;
                tD = c;
                default_case = SH_EE_A_CD;
            } else {
                tN = e + b;
                tD = c;
                default_case = SH_EE_B_CD;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0)      return SH_EE_A_C;
        else if (-d >= a)   return SH_EE_B_C;
        else                return SH_EE_AB_C;
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0)    return SH_EE_A_D;
        else if ((-d + b) >= a) return SH_EE_B_D;
        else                    return SH_EE_AB_D;
    }

    return default_case;
}

// ===========================================================================
// decidePointTriangleDistanceType (inline 2x2 edge projections)
// ===========================================================================
SH_INLINE void shPtEdgeTest(real3 from, real3 to, real3 pt,
                            real3 normal, SH_OUT(real) s, SH_OUT(real) t) {
    real3 e = to - from;
    real3 n = shCross(e, normal);
    // 2x2 system [[e.e, e.n],[n.e, n.n]] * [s,t] = [e.rhs, n.rhs]
    real a00 = shDot(e, e), a01 = shDot(e, n);
    real a10 = a01,         a11 = shDot(n, n);
    real b0  = shDot(e, pt - from), b1 = shDot(n, pt - from);
    real det = a00 * a11 - a01 * a10;
    s = (a11 * b0 - a01 * b1) / det;
    t = (a00 * b1 - a10 * b0) / det;
}

SH_INLINE int shDecidePointTriangleDistanceType(
    real3 p, real3 t0, real3 t1, real3 t2) {
    real3 normal = shCross(t1 - t0, t2 - t0);

    real s0, t0v;
    shPtEdgeTest(t0, t1, p, normal, s0, t0v);
    if (s0 > 0.0 && s0 < 1.0 && t0v >= 0.0) return SH_PT_P_AB;

    real s1, t1v;
    shPtEdgeTest(t1, t2, p, normal, s1, t1v);
    if (s1 > 0.0 && s1 < 1.0 && t1v >= 0.0) return SH_PT_P_BC;

    real s2, t2v;
    shPtEdgeTest(t2, t0, p, normal, s2, t2v);
    if (s2 > 0.0 && s2 < 1.0 && t2v >= 0.0) return SH_PT_P_CA;

    if (s0 <= 0.0 && s2 >= 1.0) return SH_PT_P_A;
    if (s1 <= 0.0 && s0 >= 1.0) return SH_PT_P_B;
    if (s2 <= 0.0 && s1 >= 1.0) return SH_PT_P_C;

    return SH_PT_P_ABC;
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_TYPE_H_
