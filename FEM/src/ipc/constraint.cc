//
// IPC Constraint — GIPC barrier 实现
//
// 核心路径:
//   1. computePFPx_XX → 得到 PFPx, q0, I5
//   2. gradient = kappa * PFPx^T * (gradCoeff(I5) * sqrt(I5) * q0)
//   3. hessian  = sandwichRank1(PFPx, q0, kappa * clampedLambda0(I5))
//
// EE mollifier 分支走 computeMollifiedBarrier (完整能量/梯度/Hessian)
//

#include <fem/ipc/constraint.h>
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/gipc/pfpx.h>
#include <fem/ipc/gipc/hessian.h>
#include <fem/ipc/gipc/mollifier.h>
#include <fem/shared/gipc-mollifier.h>   // C++/HLSL single-source EE mollifier
#include <Maths/block-types.h>
#include <glm/geometric.hpp>
#include <cassert>
#include <cmath>

namespace sim::fem::ipc {
using maths::assembleLocalGrad;
using maths::assembleLocalHessian;
using namespace gipc;

#define SIM_ASSERT_FINITE_SCALAR(x) assert(std::isfinite(x))
#define SIM_ASSERT_FINITE_EIGEN(x) assert((x).allFinite())
#define SIM_ASSERT_FINITE_VEC3(x) \
  assert(std::isfinite((x)[0]) && std::isfinite((x)[1]) && std::isfinite((x)[2]))



// =========================================================================
// VertexTriangleConstraint
// =========================================================================

void VertexTriangleConstraint::updateDistanceType() {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  type = decidePointTriangleDistanceType(p, a, b, c);
}

Real VertexTriangleConstraint::distanceSqr() const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  switch (type) {
    case PointTriangleDistanceType::P_A:   return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:   return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:   return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:  return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:  return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:  return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC: return distanceSqrPointPlane(p, a, b, c);
    default: throw std::runtime_error("Unknown distance type");
  }
}

Real VertexTriangleConstraint::barrierEnergy(const Barrier& barrier, Real kappa) const {
  Real dSqr = distanceSqr();
  return kappa * barrier.energy(dSqr);
}

void VertexTriangleConstraint::assembleBarrierGradient(
    const Barrier& barrier,
    maths::BlockVector<3>& globalGradient,
    Real kappa) const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  Real dHat = barrier.dHat();

  SIM_ASSERT_FINITE_VEC3(p);
  SIM_ASSERT_FINITE_VEC3(a);
  SIM_ASSERT_FINITE_VEC3(b);
  SIM_ASSERT_FINITE_VEC3(c);
  SIM_ASSERT_FINITE_SCALAR(dHat);
  SIM_ASSERT_FINITE_SCALAR(kappa);

  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto pfpx = computePFPx_PP(p, a, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[0]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      auto pfpx = computePFPx_PP(p, b, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      auto pfpx = computePFPx_PP(p, c, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[2]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      auto pfpx = computePFPx_PE(p, a, b, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      SIM_ASSERT_FINITE_VEC3(localGrad[2]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      auto pfpx = computePFPx_PE(p, b, c, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      SIM_ASSERT_FINITE_VEC3(localGrad[2]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[1], globalTriVerts[2]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      auto pfpx = computePFPx_PE(p, c, a, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      SIM_ASSERT_FINITE_VEC3(localGrad[2]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[2], globalTriVerts[0]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      auto pfpx = computePFPx_PT(p, a, b, c, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 9, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 12, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto localGrad = eigenToLocalGrad<4>(grad);
      SIM_ASSERT_FINITE_VEC3(localGrad[0]);
      SIM_ASSERT_FINITE_VEC3(localGrad[1]);
      SIM_ASSERT_FINITE_VEC3(localGrad[2]);
      SIM_ASSERT_FINITE_VEC3(localGrad[3]);
      std::array<int, 4> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1], globalTriVerts[2]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[3]]);
      assembleLocalGrad<4>(globalGradient, idx, localGrad);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[3]]);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in VT gradient");
  }
}


void VertexTriangleConstraint::assembleBarrierHessian(
    const Barrier& barrier,
    maths::BlockSparseMatrix<3>& globalHessian,
    Real kappa) const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  Real dHat = barrier.dHat();

  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto pfpx = computePFPx_PP(p, a, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[0]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      auto pfpx = computePFPx_PP(p, b, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[1]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      auto pfpx = computePFPx_PP(p, c, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[2]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      auto pfpx = computePFPx_PE(p, a, b, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      auto pfpx = computePFPx_PE(p, b, c, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[1], globalTriVerts[2]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      auto pfpx = computePFPx_PE(p, c, a, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[2], globalTriVerts[0]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      auto pfpx = computePFPx_PT(p, a, b, c, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 4> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1], globalTriVerts[2]};
      assembleLocalHessian<4>(globalHessian, idx, localH);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in VT hessian");
  }
}

// =========================================================================
// EdgeEdgeConstraint
// =========================================================================

void EdgeEdgeConstraint::updateDistanceType() {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;
  auto ea0 = x[globalEdgeA[0]];
  auto ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]];
  auto eb1 = x[globalEdgeB[1]];

  const glm::dvec3 u = ea1 - ea0;
  const glm::dvec3 v = eb1 - eb0;
  const glm::dvec3 w = ea0 - eb0;

  Real a = glm::dot(u, u);
  Real b = glm::dot(u, v);
  Real c = glm::dot(v, v);
  Real d = glm::dot(u, w);
  Real e = glm::dot(v, w);
  Real D = a * c - b * b;

  if (a == 0.0 && c == 0.0) { type = EdgeEdgeDistanceType::A_C; return; }
  else if (a == 0.0) { type = EdgeEdgeDistanceType::A_CD; return; }
  else if (c == 0.0) { type = EdgeEdgeDistanceType::AB_C; return; }

  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
    type = decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
    return;
  }

  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
  Real sN = (b * e - c * d);
  double tN, tD;
  if (sN <= 0.0) {
    tN = e; tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= D) {
    tN = e + b; tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = (a * e - b * d); tD = D;
    if (tN > 0.0 && tN < tD &&
        glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
      if (sN < D / 2) { tN = e; tD = c; default_case = EdgeEdgeDistanceType::A_CD; }
      else { tN = e + b; tD = c; default_case = EdgeEdgeDistanceType::B_CD; }
    }
  }

  if (tN <= 0.0) {
    if (-d <= 0.0) type = EdgeEdgeDistanceType::A_C;
    else if (-d >= a) type = EdgeEdgeDistanceType::B_C;
    else type = EdgeEdgeDistanceType::AB_C;
  } else if (tN >= tD) {
    if ((-d + b) <= 0.0) type = EdgeEdgeDistanceType::A_D;
    else if ((-d + b) >= a) type = EdgeEdgeDistanceType::B_D;
    else type = EdgeEdgeDistanceType::AB_D;
  } else {
    type = default_case;
  }
}

Real EdgeEdgeConstraint::distanceSqr() const {
  auto A = x[globalEdgeA[0]];
  auto B = x[globalEdgeA[1]];
  auto C = x[globalEdgeB[0]];
  auto D = x[globalEdgeB[1]];
  switch (type) {
    case EdgeEdgeDistanceType::A_C:   return distanceSqrPointPoint(A, C);
    case EdgeEdgeDistanceType::A_D:   return distanceSqrPointPoint(A, D);
    case EdgeEdgeDistanceType::B_C:   return distanceSqrPointPoint(B, C);
    case EdgeEdgeDistanceType::B_D:   return distanceSqrPointPoint(B, D);
    case EdgeEdgeDistanceType::AB_C:  return distanceSqrPointLine(A, B, C);
    case EdgeEdgeDistanceType::AB_D:  return distanceSqrPointLine(A, B, D);
    case EdgeEdgeDistanceType::A_CD:  return distanceSqrPointLine(A, C, D);
    case EdgeEdgeDistanceType::B_CD:  return distanceSqrPointLine(B, C, D);
    case EdgeEdgeDistanceType::AB_CD: return distanceSqrLineLine(A, B, C, D);
    default: throw std::runtime_error("Unknown distance type in EE constraint");
  }
}

bool EdgeEdgeConstraint::usesMollifier() const {
  auto ea0 = x[globalEdgeA[0]], ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]], eb1 = x[globalEdgeB[1]];
  auto rest_ea0 = X[globalEdgeA[0]], rest_ea1 = X[globalEdgeA[1]];
  auto rest_eb0 = X[globalEdgeB[0]], rest_eb1 = X[globalEdgeB[1]];
  return needsMollifier(ea0, ea1, eb0, eb1, rest_ea0, rest_ea1, rest_eb0, rest_eb1);
}

Real EdgeEdgeConstraint::barrierEnergy(const Barrier& barrier, Real kappa) const {
  auto ea0 = x[globalEdgeA[0]], ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]], eb1 = x[globalEdgeB[1]];
  Real dSqr = distanceSqr();
  if (dSqr >= barrier.dHatSqr()) return 0.0;

  if (usesMollifier()) {
    auto rest_ea0 = X[globalEdgeA[0]], rest_ea1 = X[globalEdgeA[1]];
    auto rest_eb0 = X[globalEdgeB[0]], rest_eb1 = X[globalEdgeB[1]];
    return computeMollifiedBarrierEnergy(
        ea0, ea1, eb0, eb1,
        rest_ea0, rest_ea1, rest_eb0, rest_eb1,
        dSqr, barrier, kappa);
  }

  // 非 mollifier 分支：根据 distance type 走不同的 GIPC PFPx
  return kappa * barrier.energy(dSqr);
}

void EdgeEdgeConstraint::assembleBarrierGradient(
    const Barrier& barrier,
    maths::BlockVector<3>& globalGradient,
    Real kappa) const {
  auto ea0 = x[globalEdgeA[0]], ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]], eb1 = x[globalEdgeB[1]];
  Real dHat = barrier.dHat();
  std::array<int, 4> idx4 = {globalEdgeA[0], globalEdgeA[1],
                             globalEdgeB[0], globalEdgeB[1]};

  SIM_ASSERT_FINITE_VEC3(ea0);
  SIM_ASSERT_FINITE_VEC3(ea1);
  SIM_ASSERT_FINITE_VEC3(eb0);
  SIM_ASSERT_FINITE_VEC3(eb1);
  SIM_ASSERT_FINITE_SCALAR(dHat);
  SIM_ASSERT_FINITE_SCALAR(kappa);

  // Mollifier 分支: 近平行 EE
  if (usesMollifier()) {
    auto pfpx = computePFPx_PEE(ea0, ea1, eb0, eb1, dHat);
    if (!pfpx.valid) return;
    SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);

    Real dSqr = distanceSqr();
    SIM_ASSERT_FINITE_SCALAR(dSqr);
    if (dSqr >= barrier.dHatSqr()) return;

    auto rest_ea0 = X[globalEdgeA[0]], rest_ea1 = X[globalEdgeA[1]];
    auto rest_eb0 = X[globalEdgeB[0]], rest_eb1 = X[globalEdgeB[1]];
    SIM_ASSERT_FINITE_VEC3(rest_ea0);
    SIM_ASSERT_FINITE_VEC3(rest_ea1);
    SIM_ASSERT_FINITE_VEC3(rest_eb0);
    SIM_ASSERT_FINITE_VEC3(rest_eb1);

    auto result = computeMollifiedBarrier(
        ea0, ea1, eb0, eb1,
        rest_ea0, rest_ea1, rest_eb0, rest_eb1,
        dSqr, pfpx.PFPx, barrier, kappa);
    if (!result.active) return;

    SIM_ASSERT_FINITE_SCALAR(result.energy);
    SIM_ASSERT_FINITE_VEC3(result.gradient[0]);
    SIM_ASSERT_FINITE_VEC3(result.gradient[1]);
    SIM_ASSERT_FINITE_VEC3(result.gradient[2]);
    SIM_ASSERT_FINITE_VEC3(result.gradient[3]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[0]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[1]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[2]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[3]]);
    assembleLocalGrad<4>(globalGradient, idx4, result.gradient);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[0]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[1]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[2]]);
    SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[3]]);
    return;
  }

  // 非 mollifier 分支: 根据 distance type 选择 PP/PE/EE
  auto A = ea0, B = ea1, C = eb0, D = eb1;
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      auto pfpx = computePFPx_PP(A, C, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      std::array<int, 2> idx = {globalEdgeA[0], globalEdgeB[0]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case EdgeEdgeDistanceType::A_D: {
      auto pfpx = computePFPx_PP(A, D, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      std::array<int, 2> idx = {globalEdgeA[0], globalEdgeB[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case EdgeEdgeDistanceType::B_C: {
      auto pfpx = computePFPx_PP(B, C, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      std::array<int, 2> idx = {globalEdgeA[1], globalEdgeB[0]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case EdgeEdgeDistanceType::B_D: {
      auto pfpx = computePFPx_PP(B, D, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<2>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      std::array<int, 2> idx = {globalEdgeA[1], globalEdgeB[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      assembleLocalGrad<2>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      return;
    }
    case EdgeEdgeDistanceType::AB_C: {
      auto pfpx = computePFPx_PE(C, A, B, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      SIM_ASSERT_FINITE_VEC3(lg[2]);
      std::array<int, 3> idx = {globalEdgeB[0], globalEdgeA[0], globalEdgeA[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case EdgeEdgeDistanceType::AB_D: {
      auto pfpx = computePFPx_PE(D, A, B, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      SIM_ASSERT_FINITE_VEC3(lg[2]);
      std::array<int, 3> idx = {globalEdgeB[1], globalEdgeA[0], globalEdgeA[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case EdgeEdgeDistanceType::A_CD: {
      auto pfpx = computePFPx_PE(A, C, D, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      SIM_ASSERT_FINITE_VEC3(lg[2]);
      std::array<int, 3> idx = {globalEdgeA[0], globalEdgeB[0], globalEdgeB[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case EdgeEdgeDistanceType::B_CD: {
      auto pfpx = computePFPx_PE(B, C, D, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<3>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      SIM_ASSERT_FINITE_VEC3(lg[2]);
      std::array<int, 3> idx = {globalEdgeA[1], globalEdgeB[0], globalEdgeB[1]};
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      assembleLocalGrad<3>(globalGradient, idx, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx[2]]);
      return;
    }
    case EdgeEdgeDistanceType::AB_CD: {
      auto pfpx = computePFPx_EE(ea0, ea1, eb0, eb1, dHat);
      if (!pfpx.valid) return;
      SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
      SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
      SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
      if (pfpx.I5 >= 1.0) return;
      assert(pfpx.I5 > 0.0);
      Real alpha = barrier.gradCoeff(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(alpha);
      Real sqrtI5 = std::sqrt(pfpx.I5);
      SIM_ASSERT_FINITE_SCALAR(sqrtI5);
      Eigen::Matrix<Real, 9, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
      SIM_ASSERT_FINITE_EIGEN(pk1);
      Eigen::Matrix<Real, 12, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      SIM_ASSERT_FINITE_EIGEN(grad);
      auto lg = eigenToLocalGrad<4>(grad);
      SIM_ASSERT_FINITE_VEC3(lg[0]);
      SIM_ASSERT_FINITE_VEC3(lg[1]);
      SIM_ASSERT_FINITE_VEC3(lg[2]);
      SIM_ASSERT_FINITE_VEC3(lg[3]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[2]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[3]]);
      assembleLocalGrad<4>(globalGradient, idx4, lg);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[0]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[1]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[2]]);
      SIM_ASSERT_FINITE_VEC3(globalGradient[idx4[3]]);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in EE gradient");
  }
}


void EdgeEdgeConstraint::assembleBarrierHessian(
    const Barrier& barrier,
    maths::BlockSparseMatrix<3>& globalHessian,
    Real kappa) const {
  auto ea0 = x[globalEdgeA[0]], ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]], eb1 = x[globalEdgeB[1]];
  Real dHat = barrier.dHat();
  std::array<int, 4> idx4 = {globalEdgeA[0], globalEdgeA[1],
                              globalEdgeB[0], globalEdgeB[1]};

  // Mollifier 分支
  if (usesMollifier()) {
    auto pfpx = computePFPx_PEE(ea0, ea1, eb0, eb1, dHat);
    if (!pfpx.valid) return;
    Real dSqr = distanceSqr();
    if (dSqr >= barrier.dHatSqr()) return;

    auto rest_ea0 = X[globalEdgeA[0]], rest_ea1 = X[globalEdgeA[1]];
    auto rest_eb0 = X[globalEdgeB[0]], rest_eb1 = X[globalEdgeB[1]];
    auto result = computeMollifiedBarrier(
        ea0, ea1, eb0, eb1,
        rest_ea0, rest_ea1, rest_eb0, rest_eb1,
        dSqr, pfpx.PFPx, barrier, kappa);
    if (!result.active) return;
    assembleLocalHessian<4>(globalHessian, idx4, result.hessian);
    return;
  }

  // 非 mollifier 分支
  auto A = ea0, B = ea1, C = eb0, D = eb1;
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      auto pfpx = computePFPx_PP(A, C, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalEdgeA[0], globalEdgeB[0]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::A_D: {
      auto pfpx = computePFPx_PP(A, D, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalEdgeA[0], globalEdgeB[1]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::B_C: {
      auto pfpx = computePFPx_PP(B, C, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalEdgeA[1], globalEdgeB[0]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::B_D: {
      auto pfpx = computePFPx_PP(B, D, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {globalEdgeA[1], globalEdgeB[1]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::AB_C: {
      auto pfpx = computePFPx_PE(C, A, B, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalEdgeB[0], globalEdgeA[0], globalEdgeA[1]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::AB_D: {
      auto pfpx = computePFPx_PE(D, A, B, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalEdgeB[1], globalEdgeA[0], globalEdgeA[1]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::A_CD: {
      auto pfpx = computePFPx_PE(A, C, D, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalEdgeA[0], globalEdgeB[0], globalEdgeB[1]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::B_CD: {
      auto pfpx = computePFPx_PE(B, C, D, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {globalEdgeA[1], globalEdgeB[0], globalEdgeB[1]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      return;
    }
    case EdgeEdgeDistanceType::AB_CD: {
      auto pfpx = computePFPx_EE(ea0, ea1, eb0, eb1, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
      assembleLocalHessian<4>(globalHessian, idx4, localH);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in EE hessian");
  }
}

// =========================================================================
// DeformableKinematicVTConstraint — GIPC 版本
// =========================================================================

Real DeformableKinematicVTConstraint::barrierEnergy(
    const Barrier& barrier, Real kappa) const {
  Real dSqr = distanceSqr();
  return kappa * barrier.energy(dSqr);
}

void DeformableKinematicVTConstraint::assembleBarrierGradient(
    const Barrier& barrier,
    maths::BlockVector<3>& grad, Real kappa) const {
  // 对于运动学体 VT 约束，仅弹性顶点 p 有 DOF
  // GIPC 中 PP 分支: PFPx 的前 3 列对应 p0，后 3 列对应 p1
  // 我们只取 p 对应的梯度分量
  auto p = x[deformableVertex];
  Real dHat = barrier.dHat();

  SIM_ASSERT_FINITE_VEC3(p);
  SIM_ASSERT_FINITE_VEC3(ka);
  SIM_ASSERT_FINITE_VEC3(kb);
  SIM_ASSERT_FINITE_VEC3(kc);
  SIM_ASSERT_FINITE_SCALAR(dHat);
  SIM_ASSERT_FINITE_SCALAR(kappa);

  // 简化处理: 直接用 PP 距离 (运动学体只算顶点 vs 三角形最近点)
  // 计算最近点位置
  Real dSqr = distanceSqr();
  SIM_ASSERT_FINITE_SCALAR(dSqr);
  if (dSqr >= barrier.dHatSqr()) return;

  // 用 distance type 选择最近特征对应的最近点
  glm::dvec3 closest;
  switch (type) {
    case PointTriangleDistanceType::P_A: closest = ka; break;
    case PointTriangleDistanceType::P_B: closest = kb; break;
    case PointTriangleDistanceType::P_C: closest = kc; break;
    case PointTriangleDistanceType::P_AB: {
      glm::dvec3 e = kb - ka;
      SIM_ASSERT_FINITE_VEC3(e);
      Real denom = glm::dot(e, e);
      SIM_ASSERT_FINITE_SCALAR(denom);
      assert(denom > 0.0);
      Real t = std::clamp(glm::dot(p - ka, e) / denom, 0.0, 1.0);
      SIM_ASSERT_FINITE_SCALAR(t);
      closest = ka + t * e;
      break;
    }
    case PointTriangleDistanceType::P_BC: {
      glm::dvec3 e = kc - kb;
      SIM_ASSERT_FINITE_VEC3(e);
      Real denom = glm::dot(e, e);
      SIM_ASSERT_FINITE_SCALAR(denom);
      assert(denom > 0.0);
      Real t = std::clamp(glm::dot(p - kb, e) / denom, 0.0, 1.0);
      SIM_ASSERT_FINITE_SCALAR(t);
      closest = kb + t * e;
      break;
    }
    case PointTriangleDistanceType::P_CA: {
      glm::dvec3 e = ka - kc;
      SIM_ASSERT_FINITE_VEC3(e);
      Real denom = glm::dot(e, e);
      SIM_ASSERT_FINITE_SCALAR(denom);
      assert(denom > 0.0);
      Real t = std::clamp(glm::dot(p - kc, e) / denom, 0.0, 1.0);
      SIM_ASSERT_FINITE_SCALAR(t);
      closest = kc + t * e;
      break;
    }
    case PointTriangleDistanceType::P_ABC: {
      glm::dvec3 n = glm::cross(kb - ka, kc - ka);
      SIM_ASSERT_FINITE_VEC3(n);
      Real n2 = glm::dot(n, n);
      SIM_ASSERT_FINITE_SCALAR(n2);
      assert(n2 > 0.0);
      closest = p - glm::dot(p - ka, n) / n2 * n;
      break;
    }
    default: return;
  }

  SIM_ASSERT_FINITE_VEC3(closest);

  // 使用 PP GIPC: p vs closest (closest 是固定点，不参与 DOF)
  auto pfpx = computePFPx_PP(p, closest, dHat);
  if (!pfpx.valid) return;
  SIM_ASSERT_FINITE_SCALAR(pfpx.I5);
  SIM_ASSERT_FINITE_EIGEN(pfpx.PFPx);
  SIM_ASSERT_FINITE_EIGEN(pfpx.q0);
  if (pfpx.I5 >= 1.0) return;
  assert(pfpx.I5 > 0.0);
  Real alpha = barrier.gradCoeff(pfpx.I5);
  SIM_ASSERT_FINITE_SCALAR(alpha);
  Real sqrtI5 = std::sqrt(pfpx.I5);
  SIM_ASSERT_FINITE_SCALAR(sqrtI5);
  Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (alpha * sqrtI5);
  SIM_ASSERT_FINITE_EIGEN(pk1);
  Eigen::Matrix<Real, 6, 1> fullGrad = kappa * pfpx.PFPx.transpose() * pk1;
  SIM_ASSERT_FINITE_EIGEN(fullGrad);

  // 只取 p 对应的前 3 个分量 (PP PFPx 中 x = [p0; p1]，p0 是我们的弹性顶点)
  glm::dvec3 localGrad(fullGrad(0), fullGrad(1), fullGrad(2));
  SIM_ASSERT_FINITE_VEC3(localGrad);
  SIM_ASSERT_FINITE_VEC3(grad[deformableVertex]);
  grad[deformableVertex] += localGrad;
  SIM_ASSERT_FINITE_VEC3(grad[deformableVertex]);
}


void DeformableKinematicVTConstraint::assembleBarrierHessian(
    const Barrier& barrier,
    maths::BlockSparseMatrix<3>& H, Real kappa) const {
  auto p = x[deformableVertex];
  Real dHat = barrier.dHat();
  Real dSqr = distanceSqr();
  if (dSqr >= barrier.dHatSqr()) return;

  // 同样计算最近点
  glm::dvec3 closest;
  switch (type) {
    case PointTriangleDistanceType::P_A: closest = ka; break;
    case PointTriangleDistanceType::P_B: closest = kb; break;
    case PointTriangleDistanceType::P_C: closest = kc; break;
    case PointTriangleDistanceType::P_AB: {
      glm::dvec3 e = kb - ka;
      Real t = std::clamp(glm::dot(p - ka, e) / glm::dot(e, e), 0.0, 1.0);
      closest = ka + t * e;
      break;
    }
    case PointTriangleDistanceType::P_BC: {
      glm::dvec3 e = kc - kb;
      Real t = std::clamp(glm::dot(p - kb, e) / glm::dot(e, e), 0.0, 1.0);
      closest = kb + t * e;
      break;
    }
    case PointTriangleDistanceType::P_CA: {
      glm::dvec3 e = ka - kc;
      Real t = std::clamp(glm::dot(p - kc, e) / glm::dot(e, e), 0.0, 1.0);
      closest = kc + t * e;
      break;
    }
    case PointTriangleDistanceType::P_ABC: {
      glm::dvec3 n = glm::cross(kb - ka, kc - ka);
      Real n2 = glm::dot(n, n);
      closest = p - glm::dot(p - ka, n) / n2 * n;
      break;
    }
    default: return;
  }

  auto pfpx = computePFPx_PP(p, closest, dHat);
  if (!pfpx.valid || pfpx.I5 >= 1.0) return;
  Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);

  // 只取 [0][0] block: ∂²E/∂p∂p (弹性顶点对自己)
  H.addBlock(deformableVertex, deformableVertex, localH[0][0]);
}

#undef SIM_ASSERT_FINITE_VEC3
#undef SIM_ASSERT_FINITE_EIGEN
#undef SIM_ASSERT_FINITE_SCALAR

// =========================================================================
// Phase 3: 统一 Barrier 接口实现
// =========================================================================

namespace {
/// 从统一的 ConstraintPair 索引中提取几何数据

Real computePPDistanceSqr(const ConstraintPair& pair, const maths::BlockVector<3>& x) {
  return distanceSqrPointPoint(x[pair.indices[0]], x[pair.indices[1]]);
}

Real computePEDistanceSqr(const ConstraintPair& pair, const maths::BlockVector<3>& x) {
  return distanceSqrPointLine(x[pair.indices[0]], x[pair.indices[1]], x[pair.indices[2]]);
}

Real computePTDistanceSqr(const ConstraintPair& pair, const maths::BlockVector<3>& x) {
  return distanceSqrPointTriangle(x[pair.indices[0]], x[pair.indices[1]], 
                                   x[pair.indices[2]], x[pair.indices[3]]);
}

Real computeEEDistanceSqr(const ConstraintPair& pair, const maths::BlockVector<3>& x) {
  return distanceSqrLineLine(x[pair.indices[0]], x[pair.indices[1]],
                             x[pair.indices[2]], x[pair.indices[3]]);
}
}

Real constraintPairBarrierEnergy(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,
    const Barrier& barrier,
    Real kappa) {
  Real dSqr = 0.0;
  switch (pair.type) {
    case ConstraintKind::PP: dSqr = computePPDistanceSqr(pair, x); break;
    case ConstraintKind::PE: dSqr = computePEDistanceSqr(pair, x); break;
    case ConstraintKind::PT: dSqr = computePTDistanceSqr(pair, x); break;
    case ConstraintKind::EE: dSqr = computeEEDistanceSqr(pair, x); break;
    default: return 0.0;
  }
  // Near-parallel edge-edge -> mollified barrier (C1-continuous). Single source
  // of truth shared with the GPU in <fem/shared/gipc-mollifier.h>.
  if (pair.type == ConstraintKind::EE) {
    auto a0 = x[pair.indices[0]], a1 = x[pair.indices[1]];
    auto b0 = x[pair.indices[2]], b1 = x[pair.indices[3]];
    auto ra0 = X[pair.indices[0]], ra1 = X[pair.indices[1]];
    auto rb0 = X[pair.indices[2]], rb1 = X[pair.indices[3]];
    if (shared::shEEUsesMollifier(a0, a1, b0, b1, ra0, ra1, rb0, rb1))
      return shared::shMollifiedEnergyEE(a0, a1, b0, b1, ra0, ra1, rb0, rb1,
                                         barrier.dHat(), kappa);
  }
  if (dSqr >= barrier.dHatSqr()) return 0.0;
  return kappa * barrier.energy(dSqr);
}

void constraintPairBarrierGradient(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,
    maths::BlockVector<3>& globalGradient,
    const Barrier& barrier,
    Real kappa) {
  Real dHat = barrier.dHat();
  
  switch (pair.type) {
    case ConstraintKind::PP: {
      auto p1 = x[pair.indices[0]], p2 = x[pair.indices[1]];
      Real dSqr = distanceSqrPointPoint(p1, p2);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PP(p1, p2, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real sqrtI5 = std::sqrt(pfpx.I5);
      Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (barrier.gradCoeff(pfpx.I5) * sqrtI5);
      Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      auto lg = eigenToLocalGrad<2>(grad);
      std::array<int, 2> idx = {pair.indices[0], pair.indices[1]};
      assembleLocalGrad<2>(globalGradient, idx, lg);
      break;
    }
    case ConstraintKind::PE: {
      auto pt = x[pair.indices[0]], p0 = x[pair.indices[1]], p1 = x[pair.indices[2]];
      Real dSqr = distanceSqrPointLine(pt, p0, p1);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PE(pt, p0, p1, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real sqrtI5 = std::sqrt(pfpx.I5);
      Eigen::Matrix<Real, 6, 1> pk1 = pfpx.q0 * (barrier.gradCoeff(pfpx.I5) * sqrtI5);
      Eigen::Matrix<Real, 9, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      auto lg = eigenToLocalGrad<3>(grad);
      std::array<int, 3> idx = {pair.indices[0], pair.indices[1], pair.indices[2]};
      assembleLocalGrad<3>(globalGradient, idx, lg);
      break;
    }
    case ConstraintKind::PT: {
      auto pt = x[pair.indices[0]], p0 = x[pair.indices[1]], p1 = x[pair.indices[2]], p2 = x[pair.indices[3]];
      Real dSqr = distanceSqrPointTriangle(pt, p0, p1, p2);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PT(pt, p0, p1, p2, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real sqrtI5 = std::sqrt(pfpx.I5);
      Eigen::Matrix<Real, 9, 1> pk1 = pfpx.q0 * (barrier.gradCoeff(pfpx.I5) * sqrtI5);
      Eigen::Matrix<Real, 12, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      auto lg = eigenToLocalGrad<4>(grad);
      std::array<int, 4> idx = {pair.indices[0], pair.indices[1], pair.indices[2], pair.indices[3]};
      assembleLocalGrad<4>(globalGradient, idx, lg);
      break;
    }
    case ConstraintKind::EE: {
      auto a0 = x[pair.indices[0]], a1 = x[pair.indices[1]], b0 = x[pair.indices[2]], b1 = x[pair.indices[3]];
      Real dSqr = distanceSqrLineLine(a0, a1, b0, b1);
      if (dSqr >= barrier.dHatSqr()) return;
      std::array<int, 4> idx = {pair.indices[0], pair.indices[1], pair.indices[2], pair.indices[3]};
      // Near-parallel -> mollified branch (shared single source of truth).
      auto ra0 = X[pair.indices[0]], ra1 = X[pair.indices[1]];
      auto rb0 = X[pair.indices[2]], rb1 = X[pair.indices[3]];
      if (shared::shEEUsesMollifier(a0, a1, b0, b1, ra0, ra1, rb0, rb1)) {
        auto m = shared::shMollifiedBarrierEE(a0, a1, b0, b1, ra0, ra1, rb0, rb1,
                                              barrier.dHat(), kappa);
        if (!m.active) return;
        for (int v = 0; v < 4; ++v)
          globalGradient[idx[v]] += glm::dvec3(m.grad[v * 3], m.grad[v * 3 + 1], m.grad[v * 3 + 2]);
        break;
      }
      auto pfpx = computePFPx_EE(a0, a1, b0, b1, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real sqrtI5 = std::sqrt(pfpx.I5);
      Eigen::Matrix<Real, 9, 1> pk1 = pfpx.q0 * (barrier.gradCoeff(pfpx.I5) * sqrtI5);
      Eigen::Matrix<Real, 12, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
      auto lg = eigenToLocalGrad<4>(grad);
      assembleLocalGrad<4>(globalGradient, idx, lg);
      break;
    }
    default: break;
  }
}

void constraintPairBarrierHessian(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,
    maths::BlockSparseMatrix<3>& globalHessian,
    const Barrier& barrier,
    Real kappa) {
  Real dHat = barrier.dHat();
  
  switch (pair.type) {
    case ConstraintKind::PP: {
      auto p1 = x[pair.indices[0]], p2 = x[pair.indices[1]];
      Real dSqr = distanceSqrPointPoint(p1, p2);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PP(p1, p2, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 2> idx = {pair.indices[0], pair.indices[1]};
      assembleLocalHessian<2>(globalHessian, idx, localH);
      break;
    }
    case ConstraintKind::PE: {
      auto pt = x[pair.indices[0]], p0 = x[pair.indices[1]], p1 = x[pair.indices[2]];
      Real dSqr = distanceSqrPointLine(pt, p0, p1);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PE(pt, p0, p1, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 3> idx = {pair.indices[0], pair.indices[1], pair.indices[2]};
      assembleLocalHessian<3>(globalHessian, idx, localH);
      break;
    }
    case ConstraintKind::PT: {
      auto pt = x[pair.indices[0]], p0 = x[pair.indices[1]], p1 = x[pair.indices[2]], p2 = x[pair.indices[3]];
      Real dSqr = distanceSqrPointTriangle(pt, p0, p1, p2);
      if (dSqr >= barrier.dHatSqr()) return;
      auto pfpx = computePFPx_PT(pt, p0, p1, p2, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
      std::array<int, 4> idx = {pair.indices[0], pair.indices[1], pair.indices[2], pair.indices[3]};
      assembleLocalHessian<4>(globalHessian, idx, localH);
      break;
    }
    case ConstraintKind::EE: {
      auto a0 = x[pair.indices[0]], a1 = x[pair.indices[1]], b0 = x[pair.indices[2]], b1 = x[pair.indices[3]];
      Real dSqr = distanceSqrLineLine(a0, a1, b0, b1);
      if (dSqr >= barrier.dHatSqr()) return;
      std::array<int, 4> idx = {pair.indices[0], pair.indices[1], pair.indices[2], pair.indices[3]};
      // Near-parallel -> mollified branch (shared single source of truth).
      auto ra0 = X[pair.indices[0]], ra1 = X[pair.indices[1]];
      auto rb0 = X[pair.indices[2]], rb1 = X[pair.indices[3]];
      if (shared::shEEUsesMollifier(a0, a1, b0, b1, ra0, ra1, rb0, rb1)) {
        auto m = shared::shMollifiedBarrierEE(a0, a1, b0, b1, ra0, ra1, rb0, rb1,
                                              barrier.dHat(), kappa);
        if (!m.active) return;
        // H_block(I,J) = Σ_k lam[k] · w[k][I]·w[k][J]ᵀ  (glm dmat3 is col-major).
        for (int I = 0; I < 4; ++I)
          for (int J = 0; J < 4; ++J) {
            glm::dmat3 blk(0.0);
            for (int k = 0; k < m.rank; ++k)
              for (int cc = 0; cc < 3; ++cc)
                for (int rr = 0; rr < 3; ++rr)
                  blk[cc][rr] += m.lam[k] * m.w[k][I * 3 + rr] * m.w[k][J * 3 + cc];
            globalHessian.addBlock(idx[I], idx[J], blk);
          }
        break;
      }
      auto pfpx = computePFPx_EE(a0, a1, b0, b1, dHat);
      if (!pfpx.valid || pfpx.I5 >= 1.0) return;
      Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
      auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
      assembleLocalHessian<4>(globalHessian, idx, localH);
      break;
    }
    default: break;
  }
}

namespace {

void resolveColliderTriangleVertices(
    const ColliderConstraintPair& pair,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    glm::dvec3& ka,
    glm::dvec3& kb,
    glm::dvec3& kc) {
  if (pair.hasEmbeddedColliderTriangle) {
    ka = pair.colliderVertices[0];
    kb = pair.colliderVertices[1];
    kc = pair.colliderVertices[2];
    return;
  }

  if (pair.colliderIndices[0] >= 0) ka = colliderTriangleVertices[pair.colliderIndices[0]];
  if (pair.colliderIndices[1] >= 0) kb = colliderTriangleVertices[pair.colliderIndices[1]];
  if (pair.colliderIndices[2] >= 0) kc = colliderTriangleVertices[pair.colliderIndices[2]];
}

} // namespace

Real colliderConstraintPairBarrierEnergy(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    const Barrier& barrier,
    Real kappa) {
  int deformVertex = pair.writableIndices[0];
  auto p = x[deformVertex];
  glm::dvec3 ka, kb, kc;
  resolveColliderTriangleVertices(pair, colliderTriangleVertices, ka, kb, kc);

  Real dSqr = 0.0;
  switch (pair.type) {
    case ConstraintKind::PP: dSqr = distanceSqrPointPoint(p, ka); break;
    case ConstraintKind::PE: dSqr = distanceSqrPointLine(p, ka, kb); break;
    case ConstraintKind::PT: dSqr = distanceSqrPointTriangle(p, ka, kb, kc); break;
    default: return 0.0;
  }
  if (dSqr >= barrier.dHatSqr()) return 0.0;
  return kappa * barrier.energy(dSqr);
}

void colliderConstraintPairBarrierGradient(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    maths::BlockVector<3>& globalGradient,
    const Barrier& barrier,
    Real kappa) {
  int deformVertex = pair.writableIndices[0];
  auto p = x[deformVertex];
  Real dHat = barrier.dHat();

  glm::dvec3 ka, kb, kc;
  resolveColliderTriangleVertices(pair, colliderTriangleVertices, ka, kb, kc);

  glm::dvec3 closest;
  Real dSqr = 0.0;

  switch (pair.type) {
    case ConstraintKind::PP: {
      dSqr = distanceSqrPointPoint(p, ka);
      closest = ka;
      break;
    }
    case ConstraintKind::PE: {
      dSqr = distanceSqrPointLine(p, ka, kb);
      glm::dvec3 e = kb - ka;
      Real t = std::clamp(glm::dot(p - ka, e) / glm::dot(e, e), 0.0, 1.0);
      closest = ka + t * e;
      break;
    }
    case ConstraintKind::PT: {
      dSqr = distanceSqrPointTriangle(p, ka, kb, kc);
      glm::dvec3 n = glm::cross(kb - ka, kc - ka);
      Real n2 = glm::dot(n, n);
      closest = p - glm::dot(p - ka, n) / n2 * n;
      break;
    }
    default: return;
  }

  if (dSqr >= barrier.dHatSqr()) return;
  auto pfpx = computePFPx_PP(p, closest, dHat);
  if (!pfpx.valid || pfpx.I5 >= 1.0) return;
  Real sqrtI5 = std::sqrt(pfpx.I5);
  Eigen::Matrix<Real, 3, 1> pk1 = pfpx.q0 * (barrier.gradCoeff(pfpx.I5) * sqrtI5);
  Eigen::Matrix<Real, 6, 1> grad = kappa * pfpx.PFPx.transpose() * pk1;
  auto lg = eigenToLocalGrad<2>(grad);
  globalGradient[deformVertex] += lg[0];
}

void colliderConstraintPairBarrierHessian(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    maths::BlockSparseMatrix<3>& globalHessian,
    const Barrier& barrier,
    Real kappa) {
  int deformVertex = pair.writableIndices[0];
  auto p = x[deformVertex];
  Real dHat = barrier.dHat();

  glm::dvec3 ka, kb, kc;
  resolveColliderTriangleVertices(pair, colliderTriangleVertices, ka, kb, kc);

  glm::dvec3 closest;
  Real dSqr = 0.0;

  switch (pair.type) {
    case ConstraintKind::PP: {
      dSqr = distanceSqrPointPoint(p, ka);
      closest = ka;
      break;
    }
    case ConstraintKind::PE: {
      dSqr = distanceSqrPointLine(p, ka, kb);
      glm::dvec3 e = kb - ka;
      Real t = std::clamp(glm::dot(p - ka, e) / glm::dot(e, e), 0.0, 1.0);
      closest = ka + t * e;
      break;
    }
    case ConstraintKind::PT: {
      dSqr = distanceSqrPointTriangle(p, ka, kb, kc);
      glm::dvec3 n = glm::cross(kb - ka, kc - ka);
      Real n2 = glm::dot(n, n);
      closest = p - glm::dot(p - ka, n) / n2 * n;
      break;
    }
    default: return;
  }

  if (dSqr >= barrier.dHatSqr()) return;
  auto pfpx = computePFPx_PP(p, closest, dHat);
  if (!pfpx.valid || pfpx.I5 >= 1.0) return;
  Real lam = kappa * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
  globalHessian.addBlock(deformVertex, deformVertex, localH[0][0]);
}


} // namespace sim::fem::ipc

