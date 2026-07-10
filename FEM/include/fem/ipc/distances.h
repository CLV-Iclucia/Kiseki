//
// Created by creeper on 6/2/24.
//

#ifndef KISEKI_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#define KISEKI_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#include <cstdint>
#include <fem/types.h>
#include <fem/ipc/external/distances.h>
#include <fem/shared/ipc-distance.h>
#include <Maths/block-types.h>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

namespace ksk::fem::ipc {

using maths::LocalGrad;
using maths::LocalHessian;
using maths::localGradFromFlat;
using maths::localHessianFromFlat;

// ============================================================================
// 距离函数（标量返回值）— glm::dvec3 接口
// 逻辑共享自 <fem/shared/ipc-distance.h>（CPU/GPU 单一真相源）。
// ============================================================================

inline Real distanceSqrPointPoint(const glm::dvec3 &p1,
                                  const glm::dvec3 &p2) {
  return shared::shDistanceSqrPointPoint(p1, p2);
}

inline Real distanceSqrPointLine(const glm::dvec3 &p,
                                 const glm::dvec3 &l1,
                                 const glm::dvec3 &l2) {
  return shared::shDistanceSqrPointLine(p, l1, l2);
}

inline Real distanceSqrPointPlane(const glm::dvec3 &p,
                                  const glm::dvec3 &p1,
                                  const glm::dvec3 &p2,
                                  const glm::dvec3 &p3) {
  return shared::shDistanceSqrPointPlane(p, p1, p2, p3);
}

inline Real distanceSqrLineLine(const glm::dvec3 &l1,
                                const glm::dvec3 &l2,
                                const glm::dvec3 &m1,
                                const glm::dvec3 &m2) {
  return shared::shDistanceSqrLineLine(l1, l2, m1, m2);
}

// ============================================================================
// 梯度函数 — 返回 LocalGrad<N>
// ============================================================================

/// Point-Point (2 vertices): grad = [2(p1-p2), -2(p1-p2)]
inline LocalGrad<2> localDistanceSqrPointPointGradient(
    const glm::dvec3 &p1, const glm::dvec3 &p2) {
  auto d = 2.0 * (p1 - p2);
  return LocalGrad<2>({d, -d});
}

/// Point-Line (3 vertices): p, l1, l2
inline LocalGrad<3> localDistanceSqrPointLineGradient(
    const glm::dvec3 &p, const glm::dvec3 &l1, const glm::dvec3 &l2) {
  double buf[9];
  autogen::point_line_distance_gradient_3D(
      p.x, p.y, p.z, l1.x, l1.y, l1.z, l2.x, l2.y, l2.z, buf);
  return localGradFromFlat<3>(buf);
}

/// Point-Plane (4 vertices): p, p1, p2, p3
inline LocalGrad<4> localDistanceSqrPointPlaneGradient(
    const glm::dvec3 &p, const glm::dvec3 &p1,
    const glm::dvec3 &p2, const glm::dvec3 &p3) {
  double buf[12];
  autogen::point_plane_distance_gradient(
      p.x, p.y, p.z, p1.x, p1.y, p1.z,
      p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, buf);
  return localGradFromFlat<4>(buf);
}

/// Line-Line (4 vertices): l1, l2, m1, m2
inline LocalGrad<4> localDistanceSqrLineLineGradient(
    const glm::dvec3 &l1, const glm::dvec3 &l2,
    const glm::dvec3 &m1, const glm::dvec3 &m2) {
  double buf[12];
  autogen::line_line_distance_gradient(
      l1.x, l1.y, l1.z, l2.x, l2.y, l2.z,
      m1.x, m1.y, m1.z, m2.x, m2.y, m2.z, buf);
  return localGradFromFlat<4>(buf);
}

// ============================================================================
// Hessian 函数 — 返回 LocalHessian<N>
// ============================================================================

/// Point-Point (2×2 blocks): H = [[2I, -2I], [-2I, 2I]]
inline LocalHessian<2> localDistanceSqrPointPointHessian(
    const glm::dvec3 &p1, const glm::dvec3 &p2) {
  auto I = glm::dmat3(2.0);   // 2 * Identity
  auto negI = glm::dmat3(-2.0);
  LocalHessian<2> H{};
  H[0][0] = I;
  H[0][1] = negI;
  H[1][0] = negI;
  H[1][1] = I;
  return H;
}

/// Point-Line (3×3 blocks)
inline LocalHessian<3> localDistanceSqrPointLineHessian(
    const glm::dvec3 &p, const glm::dvec3 &l1, const glm::dvec3 &l2) {
  double buf[81];  // 9×9 flat
  autogen::point_line_distance_hessian_3D(
      p.x, p.y, p.z, l1.x, l1.y, l1.z, l2.x, l2.y, l2.z, buf);
  return localHessianFromFlat<3>(buf);
}

/// Point-Plane (4×4 blocks)
inline LocalHessian<4> localDistanceSqrPointPlaneHessian(
    const glm::dvec3 &p, const glm::dvec3 &p1,
    const glm::dvec3 &p2, const glm::dvec3 &p3) {
  double buf[144];  // 12×12 flat
  autogen::point_plane_distance_hessian(
      p.x, p.y, p.z, p1.x, p1.y, p1.z,
      p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, buf);
  return localHessianFromFlat<4>(buf);
}

/// Line-Line (4×4 blocks)
inline LocalHessian<4> localDistanceSqrLineLineHessian(
    const glm::dvec3 &l1, const glm::dvec3 &l2,
    const glm::dvec3 &m1, const glm::dvec3 &m2) {
  double buf[144];  // 12×12 flat
  autogen::line_line_distance_hessian(
      l1.x, l1.y, l1.z, l2.x, l2.y, l2.z,
      m1.x, m1.y, m1.z, m2.x, m2.y, m2.z, buf);
  return localHessianFromFlat<4>(buf);
}

enum class EdgeEdgeDistanceType : uint8_t {
  A_C,
  A_D,
  B_C,
  B_D,
  AB_C,
  AB_D,
  A_CD,
  B_CD,
  AB_CD,
  Unknown
};

enum class PointTriangleDistanceType : uint8_t {
  P_A,
  P_B,
  P_C,
  P_AB,
  P_BC,
  P_CA,
  P_ABC,
  Unknown
};

// ============================================================================
// decideEdgeEdgeParallelDistanceType — glm::dvec3 版本
// 逻辑共享自 <fem/shared/ipc-distance-type.h>（CPU/GPU 单一真相源）。
// ============================================================================
inline EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(
    const glm::dvec3 &ea0, const glm::dvec3 &ea1,
    const glm::dvec3 &eb0, const glm::dvec3 &eb1) {
  return EdgeEdgeDistanceType(
      shared::shDecideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1));
}

// ============================================================================
// decideEdgeEdgeDistanceType — glm::dvec3 版本
// 逻辑共享自 <fem/shared/ipc-distance-type.h>（CPU/GPU 单一真相源）。
// ============================================================================
inline EdgeEdgeDistanceType decideEdgeEdgeDistanceType(
    const glm::dvec3 &ea0, const glm::dvec3 &ea1,
    const glm::dvec3 &eb0, const glm::dvec3 &eb1) {
  return EdgeEdgeDistanceType(
      shared::shDecideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1));
}

// ============================================================================
// decidePointTriangleDistanceType — glm::dvec3 版本
// 逻辑共享自 <fem/shared/ipc-distance-type.h>（CPU/GPU 单一真相源）。
// ============================================================================
inline PointTriangleDistanceType
decidePointTriangleDistanceType(const glm::dvec3 &p,
                                const glm::dvec3 &t0,
                                const glm::dvec3 &t1,
                                const glm::dvec3 &t2) {
  return PointTriangleDistanceType(
      shared::shDecidePointTriangleDistanceType(p, t0, t1, t2));
}

// ============================================================================
// 剩余函数声明 — glm::dvec3 接口
// ============================================================================

Real distanceSqrPointTriangle(const glm::dvec3 &p,
                              const glm::dvec3 &a,
                              const glm::dvec3 &b,
                              const glm::dvec3 &c);

LocalGrad<4> localDistancePointTriangleGradient(const glm::dvec3 &p,
                                                    const glm::dvec3 &a,
                                                    const glm::dvec3 &b,
                                                    const glm::dvec3 &c);

LocalHessian<4> localDistancePointTriangleHessian(
    const glm::dvec3 &p, const glm::dvec3 &a,
    const glm::dvec3 &b, const glm::dvec3 &c);

Real distanceSqrEdgeEdge(const glm::dvec3 &ea0, const glm::dvec3 &ea1,
                         const glm::dvec3 &eb0,
                         const glm::dvec3 &eb1);

LocalGrad<4> localDistanceSqrEdgeEdgeGradient(const glm::dvec3 &a,
                                                  const glm::dvec3 &b,
                                                  const glm::dvec3 &c,
                                                  const glm::dvec3 &d);

LocalHessian<4> localDistanceSqrEdgeEdgeHessian(const glm::dvec3 &a,
                                                     const glm::dvec3 &b,
                                                     const glm::dvec3 &c,
                                                     const glm::dvec3 &d);

} // namespace ksk::fem::ipc
#endif // KISEKI_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
