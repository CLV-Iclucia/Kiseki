//
// GIPC PFPx 鏋勯€犲櫒 鈥?涓烘瘡绉嶆帴瑙﹀鏋勯€犱汉閫犲舰鍙樻搴?F 鍜?鈭倂ec(F)/鈭倄
//
// 鏍稿績鎬濇兂锛圙IPC #else 璺緞锛?
//   1. 鎶婃硶鍚戞姇褰卞埌鍥哄畾鏂瑰悜锛屾瀯閫犱汉閫?Dm锛堜娇璺濈=d虃 鏃?I鈧?1锛?
//   2. F = Ds 路 Dm鈦宦?
//   3. I鈧?= ||F路n||虏 = (d/d虃)虏
//   4. PFPx = 鈭倂ec(F)/鈭倄 鏄嚎鎬у父鏁扮煩闃?
//
// 杈撳嚭鍙洿鎺ョ敤浜?GIPC 鐨?gradient sandwich 鍜?Hessian sandwich
//

#pragma once
#include <Contact/types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

namespace ksk::engine::contact::gipc {

struct PFPxResult12 {
  Eigen::Matrix<Real, 9, 12> PFPx;
  Eigen::Matrix<Real, 9, 1> q0;
  Real I5;
  bool valid = false;
};

struct PFPxResult9 {
  Eigen::Matrix<Real, 6, 9> PFPx;
  Eigen::Matrix<Real, 6, 1> q0;
  Real I5;
  bool valid = false;
};

struct PFPxResult6 {
  Eigen::Matrix<Real, 3, 6> PFPx;
  Eigen::Matrix<Real, 3, 1> q0;
  Real I5;
  bool valid = false;
};


/// Edge-Edge: x = [ea0; ea1; eb0; eb1], 12 DOF
/// Ds = [ea1-ea0 | eb0-ea0 | eb1-ea0]
PFPxResult12 computePFPx_EE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat);

PFPxResult12 computePFPx_PEE(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    Real dHat);

/// Point-Triangle: x = [p; t0; t1; t2], 12 DOF
/// Ds = [t0-p | t1-p | t2-p],
PFPxResult12 computePFPx_PT(
    const glm::dvec3& p,
    const glm::dvec3& t0, const glm::dvec3& t1, const glm::dvec3& t2,
    Real dHat);

PFPxResult9 computePFPx_PE(
    const glm::dvec3& p,
    const glm::dvec3& e0, const glm::dvec3& e1,
    Real dHat);

PFPxResult6 computePFPx_PP(
    const glm::dvec3& p0, const glm::dvec3& p1,
    Real dHat, Real dReserved = 0.0);

namespace detail {

Eigen::Matrix<Real, 9, 12> computePFPx3D(const Eigen::Matrix3d& DmInv);

Eigen::Matrix<Real, 6, 9> computePFPx3x2(const Eigen::Matrix2d& DmInv);

Eigen::Matrix<Real, 3, 6> computePFPx3x1(Real DmInv);

} // namespace detail

} // namespace ksk::engine::contact::gipc
