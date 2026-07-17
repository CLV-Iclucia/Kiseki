//
// GIPC Hessian 杈呭姪 鈥?瑙ｆ瀽 SPD 鎶曞奖 + sandwich + makePD2x2
//
// 鎻愪緵浠?GIPC 鍐呭眰 Hessian (rank-1 鎴?rank-3) 鍒?LocalHessian<N> 鐨勫畬鏁寸绾?
//

#pragma once
#include <Contact/types.h>
#include <Maths/block-types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

namespace ksk::engine::contact::gipc {

// ============================================================================
// makePD2x2: 2脳2 瀵圭О鐭╅樀鐨?SPD 鎶曞奖
// ============================================================================

struct PD2x2Result {
  Real eigenValues[2] = {0, 0};
  glm::dvec2 eigenVecs[2];
  int numPositive = 0;
};

inline PD2x2Result makePD2x2(Real a, Real b, Real d) {
  PD2x2Result result;
  Real trace = a + d;
  Real det = a * d - b * b;
  Real disc = trace * trace - 4.0 * det;
  disc = std::max(disc, 0.0);
  Real sqrtDisc = std::sqrt(disc);

  Real lam1 = 0.5 * (trace + sqrtDisc);
  Real lam2 = 0.5 * (trace - sqrtDisc);

  auto computeEigvec = [&](Real lam) -> glm::dvec2 {
    // (A - lam*I) * v = 0
    // 鍙?v = normalize([-b, a-lam]) 鎴?normalize([d-lam, -b])
    Real r0 = a - lam, r1 = b;
    if (std::abs(r0) + std::abs(r1) < 1e-30) {
      // 鐗瑰緛鍊奸€€鍖栵紝鍙栨浜ゆ柟鍚?
      r0 = -b; r1 = d - lam;
    }
    glm::dvec2 v(-r1, r0);
    Real norm = glm::length(v);
    return norm > 1e-30 ? v / norm : glm::dvec2(1, 0);
  };

  result.numPositive = 0;
  if (lam1 > 0) {
    result.eigenValues[result.numPositive] = lam1;
    result.eigenVecs[result.numPositive] = computeEigvec(lam1);
    result.numPositive++;
  }
  if (lam2 > 0) {
    result.eigenValues[result.numPositive] = lam2;
    result.eigenVecs[result.numPositive] = computeEigvec(lam2);
    result.numPositive++;
  }
  return result;
}

// ============================================================================
// Sandwich: 浠庡唴灞?Hessian 鍒?LocalHessian<N>
// ============================================================================

/// 绠€鍗曞垎鏀?sandwich (rank-1):
///   H_local(12脳12) = PFPx^T 路 (位鈧€ 路 q鈧€ 路 q鈧€^T) 路 PFPx
///   绛変环浜?H_local = 位鈧€ 路 (PFPx^T 路 q鈧€) 路 (PFPx^T 路 q鈧€)^T
/// 杈撳嚭杞崲涓?LocalHessian<4>
template<int N, int VecDim>
maths::LocalHessian<N> sandwichRank1(
    const Eigen::Matrix<Real, VecDim, N * 3>& PFPx,
    const Eigen::Matrix<Real, VecDim, 1>& q0,
    Real lambda0) {
  // v = PFPx^T 路 q鈧€ (N*3 脳 1)
  Eigen::Matrix<Real, N * 3, 1> v = PFPx.transpose() * q0;
  // H = 位鈧€ 路 v 路 v岬€ (N*3 脳 N*3)
  Eigen::Matrix<Real, N * 3, N * 3> H = lambda0 * v * v.transpose();
  // 杞崲涓?LocalHessian<N>
  return eigenToLocalHessian<N>(H);
}

/// 閫氱敤 sandwich:
///   H_local(N*3 脳 N*3) = PFPx^T 路 H_inner 路 PFPx
template<int N, int VecDim>
maths::LocalHessian<N> sandwichFull(
    const Eigen::Matrix<Real, VecDim, N * 3>& PFPx,
    const Eigen::Matrix<Real, VecDim, VecDim>& H_inner) {
  Eigen::Matrix<Real, N * 3, N * 3> H = PFPx.transpose() * H_inner * PFPx;
  return eigenToLocalHessian<N>(H);
}

// ============================================================================
// Eigen 鐭╅樀 鈫?LocalHessian<N> 杞崲
// ============================================================================

/// 灏?(N*3 脳 N*3) Eigen 鐭╅樀杞崲涓?LocalHessian<N> (glm column-major blocks)
template<int N>
maths::LocalHessian<N> eigenToLocalHessian(
    const Eigen::Matrix<Real, N * 3, N * 3>& H) {
  maths::LocalHessian<N> result{};
  for (int bi = 0; bi < N; bi++) {
    for (int bj = 0; bj < N; bj++) {
      // glm::dmat3 鏄?column-major: mat[col][row]
      for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
          result[bi][bj][c][r] = H(bi * 3 + r, bj * 3 + c);
    }
  }
  return result;
}

/// 灏?(N*3 脳 1) Eigen 鍚戦噺杞崲涓?LocalGrad<N>
template<int N>
maths::LocalGrad<N> eigenToLocalGrad(
    const Eigen::Matrix<Real, N * 3, 1>& g) {
  maths::LocalGrad<N> result{};
  for (int i = 0; i < N; i++)
    result[i] = glm::dvec3(g(i * 3), g(i * 3 + 1), g(i * 3 + 2));
  return result;
}

} // namespace ksk::engine::contact::gipc
