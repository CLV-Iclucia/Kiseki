//
// ipc-distance.h — shared (C++ + HLSL) IPC squared-distance functions.
//
// Single source of truth for the point-triangle / edge-edge squared distance
// used by the dHat activation test. The GPU activation kernel must compute the
// EXACT same value as the CPU, otherwise candidates near the dHat boundary
// would be classified differently. All math stays in full double (the shDot/
// shCross helpers avoid the float-only HLSL intrinsics).
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_H_

#include <fem/shared/ipc-distance-type.h>

SH_NS_BEGIN

// ---- elementary squared distances -----------------------------------------
SH_INLINE real shDistanceSqrPointPoint(real3 p1, real3 p2) {
    real3 d = p1 - p2;
    return shDot(d, d);
}

SH_INLINE real shDistanceSqrPointLine(real3 p, real3 l1, real3 l2) {
    real3 e = l2 - l1;
    real  e_len2 = shDot(e, e);
    SH_ASSERT(e_len2 != 0.0);
    real3 c = shCross(e, p - l1);
    return shDot(c, c) / e_len2;
}

SH_INLINE real shDistanceSqrPointPlane(real3 p, real3 p1,
                                          real3 p2, real3 p3) {
    real3 normal = shCross(p2 - p1, p3 - p1);
    real  n_len2 = shDot(normal, normal);
    SH_ASSERT(n_len2 != 0.0);
    real  d = shDot(p - p1, normal);
    return d * d / n_len2;
}

SH_INLINE real shDistanceSqrLineLine(real3 l1, real3 l2,
                                        real3 m1, real3 m2) {
    real3 normal = shCross(l2 - l1, m2 - m1);
    real  n_len2 = shDot(normal, normal);
    if (n_len2 == 0.0)
        return shDistanceSqrPointLine(l1, m1, m2);
    real line_to_line = shDot(l1 - m1, normal);
    line_to_line = line_to_line * line_to_line;
    return line_to_line / n_len2;
}

// ---- type-dispatched squared distances -------------------------------------
// `type` is a precomputed code from shDecide*DistanceType (reused so the
// activation classification and the distance use the SAME type, matching CPU).

SH_INLINE real shDistanceSqrPointTriangleByType(
    int type, real3 p, real3 a, real3 b, real3 c) {
    if (type == SH_PT_P_A)  return shDistanceSqrPointPoint(p, a);
    if (type == SH_PT_P_B)  return shDistanceSqrPointPoint(p, b);
    if (type == SH_PT_P_C)  return shDistanceSqrPointPoint(p, c);
    if (type == SH_PT_P_AB) return shDistanceSqrPointLine(p, a, b);
    if (type == SH_PT_P_BC) return shDistanceSqrPointLine(p, b, c);
    if (type == SH_PT_P_CA) return shDistanceSqrPointLine(p, c, a);
    return shDistanceSqrPointPlane(p, a, b, c);  // SH_PT_P_ABC
}

SH_INLINE real shDistanceSqrEdgeEdgeByType(
    int type, real3 ea0, real3 ea1, real3 eb0, real3 eb1) {
    if (type == SH_EE_A_C)  return shDistanceSqrPointPoint(ea0, eb0);
    if (type == SH_EE_A_D)  return shDistanceSqrPointPoint(ea0, eb1);
    if (type == SH_EE_B_C)  return shDistanceSqrPointPoint(ea1, eb0);
    if (type == SH_EE_B_D)  return shDistanceSqrPointPoint(ea1, eb1);
    if (type == SH_EE_A_CD) return shDistanceSqrPointLine(ea0, eb0, eb1);
    if (type == SH_EE_B_CD) return shDistanceSqrPointLine(ea1, eb0, eb1);
    if (type == SH_EE_AB_C) return shDistanceSqrPointLine(eb0, ea0, ea1);
    if (type == SH_EE_AB_D) return shDistanceSqrPointLine(eb1, ea0, ea1);
    return shDistanceSqrLineLine(ea0, ea1, eb0, eb1);  // SH_EE_AB_CD
}

// ---- convenience: decide + dispatch ----------------------------------------
SH_INLINE real shDistanceSqrPointTriangle(real3 p, real3 a,
                                             real3 b, real3 c) {
    return shDistanceSqrPointTriangleByType(
        shDecidePointTriangleDistanceType(p, a, b, c), p, a, b, c);
}

SH_INLINE real shDistanceSqrEdgeEdge(real3 ea0, real3 ea1,
                                        real3 eb0, real3 eb1) {
    return shDistanceSqrEdgeEdgeByType(
        shDecideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1), ea0, ea1, eb0, eb1);
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_DISTANCE_H_
