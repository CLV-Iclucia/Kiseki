//
// Created by creeper on 9/7/24.
//

#ifndef KISEKI_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_
#define KISEKI_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_

#include <glm/glm.hpp>
#include <fem/ipc/external/mollifier.h>
#include <Maths/block-types.h>

namespace ksk::fem::ipc {

using maths::LocalGrad;
using maths::LocalHessian;
using maths::localGradFromFlat;
using maths::localHessianFromFlat;

/// edgeEdgeCrossSquareNormGradient — 返回 LocalGrad<4>
inline LocalGrad<4> edgeEdgeCrossSquareNormGradient(
    const glm::dvec3 &ea0,
    const glm::dvec3 &ea1,
    const glm::dvec3 &eb0,
    const glm::dvec3 &eb1) {
  double g[12];
  autogen::edge_edge_cross_squarednorm_gradient(
      ea0.x, ea0.y, ea0.z,
      ea1.x, ea1.y, ea1.z,
      eb0.x, eb0.y, eb0.z,
      eb1.x, eb1.y, eb1.z,
      g);
  return localGradFromFlat<4>(g);
}

/// edgeEdgeCrossSquaredNormHessian — 返回 LocalHessian<4>
inline LocalHessian<4> edgeEdgeCrossSquaredNormHessian(
    const glm::dvec3 &ea0,
    const glm::dvec3 &ea1,
    const glm::dvec3 &eb0,
    const glm::dvec3 &eb1) {
  double h[144];
  autogen::edge_edge_cross_squarednorm_hessian(
      ea0.x, ea0.y, ea0.z,
      ea1.x, ea1.y, ea1.z,
      eb0.x, eb0.y, eb0.z,
      eb1.x, eb1.y, eb1.z,
      h);
  return localHessianFromFlat<4>(h);
}

/// edgeEdgeMollifierThresholdGradient — 返回 LocalGrad<4>
inline LocalGrad<4> edgeEdgeMollifierThresholdGradient(
    const glm::dvec3 &ea0,
    const glm::dvec3 &ea1,
    const glm::dvec3 &eb0,
    const glm::dvec3 &eb1,
    Real scale) {
  double g[12];
  autogen::edge_edge_mollifier_threshold_gradient(
      ea0.x, ea0.y, ea0.z,
      ea1.x, ea1.y, ea1.z,
      eb0.x, eb0.y, eb0.z,
      eb1.x, eb1.y, eb1.z,
      g, scale);
  return localGradFromFlat<4>(g);
}

} // namespace ksk::fem::ipc
#endif // KISEKI_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_
