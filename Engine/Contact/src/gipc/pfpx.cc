//
// GIPC PFPx 瀹炵幇 鈥?璋冪敤 external autogen 浠ｇ爜 (NEWF 璺緞)
//
// 鎵€鏈夊嚱鏁板唴閮ㄨ皟鐢?GIPC_PDerivative.h 涓殑绗﹀彿姹傚浠ｇ爜锛?
// 瀵瑰鏆撮湶 Eigen 绫诲瀷鎺ュ彛锛屼笉娉勯湶 external 鐨勪换浣曠被鍨嬨€?
//

#include <Contact/gipc/pfpx.h>
#include <Contact/gipc/external/GIPC_PDerivative.h>
#include <cassert>
#include <cmath>
#include <stdexcept>
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"


namespace ksk::engine::contact::gipc {

// ============================================================================
// Edge-Edge Mollifier (PEE) PFPx
// ============================================================================

PFPxResult12 computePFPx_PEE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat) {
  PFPxResult12 result;
  result.valid = false;

  assert(std::isfinite(dHat));
  assert(dHat > 0.0);
  if (l2Norm(glm::cross(ea1 - ea0, eb1 - eb0)) == 0.0) return result;

  external::Matrix12x9d raw;
  external::pFpx_pee(ea0, ea1, eb0, eb1, dHat, raw);

  // raw.m[i][j]: i=DOF index(12), j=vec(F) index(9)
  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 9; j++)
      result.PFPx(j, i) = raw.m[i][j];

  if (!result.PFPx.allFinite())
  {
    assert(false);
  }

  result.q0.setZero();
  result.I5 = 0.0;
  assert(result.q0.allFinite());
  assert(std::isfinite(result.I5));
  result.valid = true;
  return result;
}


PFPxResult12 computePFPx_EE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat) {
  PFPxResult12 result;
  result.valid = false;

  assert(std::isfinite(dHat));
  assert(dHat > 0.0);

  //  line-line  I5
  glm::dvec3 edgeA = ea1 - ea0;
  glm::dvec3 edgeB = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(edgeA, edgeB);
  Real crossLen = glm::length(crossVec);
  assert(std::isfinite(crossLen));
  if (crossLen < 1e-30) return result;
  glm::dvec3 normal = crossVec / crossLen;
  Real dis = std::abs(glm::dot(ea0 - eb0, normal));
  assert(std::isfinite(dis));
  result.I5 = (dis / dHat) * (dis / dHat);
  assert(std::isfinite(result.I5));

  external::Matrix12x9d raw;
  external::pFpx_ee2(ea0, ea1, eb0, eb1, dHat, raw);

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 9; j++)
      result.PFPx(j, i) = raw.m[i][j];

  assert(result.PFPx.allFinite());

  // NEWF: q0 = (0,...,0,1)
  result.q0.setZero();
  result.q0(8) = 1.0;
  assert(result.q0.allFinite());

  result.valid = true;
  return result;
}


// ============================================================================
// Point-Triangle PFPx
// ============================================================================

PFPxResult12 computePFPx_PT(
    const glm::dvec3& p,
    const glm::dvec3& t0, const glm::dvec3& t1, const glm::dvec3& t2,
    Real dHat) {
  PFPxResult12 result;
  result.valid = false;

  assert(std::isfinite(dHat));
  assert(dHat > 0.0);

  glm::dvec3 triNormal = glm::cross(t1 - t0, t2 - t0);
  Real triLen = glm::length(triNormal);
  assert(std::isfinite(triLen));
  if (triLen < 1e-30) return result;
  triNormal /= triLen;
  Real dis = std::abs(glm::dot(p - t0, triNormal));
  assert(std::isfinite(dis));
  result.I5 = (dis / dHat) * (dis / dHat);
  assert(std::isfinite(result.I5));

  external::Matrix12x9d raw;
  external::pFpx_pt2(p, t0, t1, t2, dHat, raw);

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 9; j++)
      result.PFPx(j, i) = raw.m[i][j];

  assert(result.PFPx.allFinite());
  result.q0.setZero();
  result.q0(8) = 1.0;
  assert(result.q0.allFinite());

  result.valid = true;
  return result;
}

// ============================================================================
// Point-Edge PFPx
//
// pFpx_pe2  Matrix9x4d PFPxT
// GIPC.cu: q0=(0,0,0,1), H4x4=lambda0*q0*q0^T
//
// PFPxResult9 (PFPx: 6x9, q0: 6x1):
//   PFPxT^T 4  PFPx(6x9)  4 q0=(0,0,0,1,0,0)
// ============================================================================

PFPxResult9 computePFPx_PE(
    const glm::dvec3& p,
    const glm::dvec3& e0, const glm::dvec3& e1,
    Real dHat) {
  PFPxResult9 result;
  result.valid = false;

  assert(std::isfinite(dHat));
  assert(dHat > 0.0);

  glm::dvec3 edgeDir = e1 - e0;
  Real edgeLen = glm::length(edgeDir);
  assert(std::isfinite(edgeLen));
  if (edgeLen < 1e-30) return result;
  glm::dvec3 tangent = edgeDir / edgeLen;
  glm::dvec3 v0 = e0 - p, v1 = e1 - p;
  glm::dvec3 triNormal = glm::cross(v0, v1);
  Real triLen = glm::length(triNormal);
  assert(std::isfinite(triLen));
  if (triLen < 1e-30) return result;
  triNormal /= triLen;
  glm::dvec3 edgeNormal = glm::cross(triNormal, tangent);
  Real dis = std::abs(glm::dot(p - e0, edgeNormal));
  assert(std::isfinite(dis));
  result.I5 = (dis / dHat) * (dis / dHat);
  assert(std::isfinite(result.I5));

  external::Matrix9x4d raw;
  external::pFpx_pe2(p, e0, e1, dHat, raw);

  // raw.m[i][j]: i=DOF(9), j=vec(F)(4)
  //  PFPx(j,i) = raw.m[i][j]
  result.PFPx.setZero();
  for (int i = 0; i < 9; i++)
    for (int j = 0; j < 4; j++)
      result.PFPx(j, i) = raw.m[i][j];

  assert(result.PFPx.allFinite());

  // q0 = (0,0,0,1,0,0)
  result.q0.setZero();
  result.q0(3) = 1.0;
  assert(result.q0.allFinite());

  result.valid = true;
  return result;
}

// ============================================================================
// GIPC.cu: Hessian_6x6 = lambda0 * v * v^T, gradient_6 = flatten_pk1 * v
//
// ============================================================================

PFPxResult6 computePFPx_PP(
    const glm::dvec3& p0, const glm::dvec3& p1,
    Real dHat, Real dReserved) {
  PFPxResult6 result;
  result.valid = false;

  assert(std::isfinite(dHat));
  assert(dHat > 0.0);

  Real dis = glm::length(p1 - p0) - dReserved;
  assert(std::isfinite(dis));
  if (dis < 1e-30) return result;
  result.I5 = (dis / dHat) * (dis / dHat);
  assert(std::isfinite(result.I5));

  external::Vector6 raw;
  external::pFpx_pp2(p0, p1, dHat, raw);

  result.PFPx.setZero();
  for (int i = 0; i < 6; i++)
    result.PFPx(2, i) = raw.v[i];

  assert(result.PFPx.allFinite());
  result.q0.setZero();
  result.q0(2) = 1.0;
  assert(result.q0.allFinite());

  result.valid = true;
  return result;
}


namespace detail {

Eigen::Matrix<Real, 9, 12> computePFPx3D(const Eigen::Matrix3d& DmInv) {
  Eigen::Matrix<Real, 9, 12> PFPx;
  PFPx.setZero();
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) {
      int vecF_idx = j * 3 + i;
      for (int k = 0; k < 3; k++) {
        Real coeff = DmInv(k, j);
        PFPx(vecF_idx, (k + 1) * 3 + i) += coeff;
        PFPx(vecF_idx, 0 * 3 + i) -= coeff;
      }
    }
  return PFPx;
}

Eigen::Matrix<Real, 6, 9> computePFPx3x2(const Eigen::Matrix2d& DmInv) {
  Eigen::Matrix<Real, 6, 9> PFPx;
  PFPx.setZero();
  for (int j = 0; j < 2; j++)
    for (int i = 0; i < 3; i++) {
      int vecF_idx = j * 3 + i;
      for (int k = 0; k < 2; k++) {
        Real coeff = DmInv(k, j);
        PFPx(vecF_idx, (k + 1) * 3 + i) += coeff;
        PFPx(vecF_idx, 0 * 3 + i) -= coeff;
      }
    }
  return PFPx;
}

Eigen::Matrix<Real, 3, 6> computePFPx3x1(Real DmInv) {
  Eigen::Matrix<Real, 3, 6> PFPx;
  PFPx.setZero();
  for (int i = 0; i < 3; i++) {
    PFPx(i, 3 + i) = DmInv;
    PFPx(i, 0 + i) = -DmInv;
  }
  return PFPx;
}

} // namespace detail

} // namespace ksk::engine::contact::gipc
