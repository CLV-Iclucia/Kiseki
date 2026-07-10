//
// GIPC Hessian 辅助 — 解析 SPD 投影 + sandwich + makePD2x2
//
// 提供从 GIPC 内层 Hessian (rank-1 或 rank-3) 到 LocalHessian<N> 的完整管线
//

#pragma once
#include <fem/types.h>
#include <Maths/block-types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

namespace ksk::fem::ipc::gipc {

// ============================================================================
// makePD2x2: 2×2 对称矩阵的 SPD 投影
// ============================================================================

struct PD2x2Result {
  Real eigenValues[2] = {0, 0};
  Eigen::Vector2d eigenVecs[2];
  int numPositive = 0;  // 正特征值个数
};

/// 2×2 对称矩阵 [[a, b],[b, d]] 的特征分解 + 只保留正特征值
inline PD2x2Result makePD2x2(Real a, Real b, Real d) {
  PD2x2Result result;
  Real trace = a + d;
  Real det = a * d - b * b;
  Real disc = trace * trace - 4.0 * det;
  disc = std::max(disc, 0.0);
  Real sqrtDisc = std::sqrt(disc);

  Real lam1 = 0.5 * (trace + sqrtDisc);
  Real lam2 = 0.5 * (trace - sqrtDisc);

  // 特征向量
  auto computeEigvec = [&](Real lam) -> Eigen::Vector2d {
    // (A - lam*I) * v = 0
    // 取 v = normalize([-b, a-lam]) 或 normalize([d-lam, -b])
    Real r0 = a - lam, r1 = b;
    if (std::abs(r0) + std::abs(r1) < 1e-30) {
      // 特征值退化，取正交方向
      r0 = -b; r1 = d - lam;
    }
    Eigen::Vector2d v(-r1, r0);
    Real norm = v.norm();
    return norm > 1e-30 ? v / norm : Eigen::Vector2d(1, 0);
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
// Sandwich: 从内层 Hessian 到 LocalHessian<N>
// ============================================================================

/// 简单分支 sandwich (rank-1):
///   H_local(12×12) = PFPx^T · (λ₀ · q₀ · q₀^T) · PFPx
///   等价于 H_local = λ₀ · (PFPx^T · q₀) · (PFPx^T · q₀)^T
/// 输出转换为 LocalHessian<4>
template<int N, int VecDim>
maths::LocalHessian<N> sandwichRank1(
    const Eigen::Matrix<Real, VecDim, N * 3>& PFPx,
    const Eigen::Matrix<Real, VecDim, 1>& q0,
    Real lambda0) {
  // v = PFPx^T · q₀ (N*3 × 1)
  Eigen::Matrix<Real, N * 3, 1> v = PFPx.transpose() * q0;
  // H = λ₀ · v · vᵀ (N*3 × N*3)
  Eigen::Matrix<Real, N * 3, N * 3> H = lambda0 * v * v.transpose();
  // 转换为 LocalHessian<N>
  return eigenToLocalHessian<N>(H);
}

/// 通用 sandwich:
///   H_local(N*3 × N*3) = PFPx^T · H_inner · PFPx
template<int N, int VecDim>
maths::LocalHessian<N> sandwichFull(
    const Eigen::Matrix<Real, VecDim, N * 3>& PFPx,
    const Eigen::Matrix<Real, VecDim, VecDim>& H_inner) {
  Eigen::Matrix<Real, N * 3, N * 3> H = PFPx.transpose() * H_inner * PFPx;
  return eigenToLocalHessian<N>(H);
}

// ============================================================================
// Eigen 矩阵 → LocalHessian<N> 转换
// ============================================================================

/// 将 (N*3 × N*3) Eigen 矩阵转换为 LocalHessian<N> (glm column-major blocks)
template<int N>
maths::LocalHessian<N> eigenToLocalHessian(
    const Eigen::Matrix<Real, N * 3, N * 3>& H) {
  maths::LocalHessian<N> result{};
  for (int bi = 0; bi < N; bi++) {
    for (int bj = 0; bj < N; bj++) {
      // glm::dmat3 是 column-major: mat[col][row]
      for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
          result[bi][bj][c][r] = H(bi * 3 + r, bj * 3 + c);
    }
  }
  return result;
}

/// 将 (N*3 × 1) Eigen 向量转换为 LocalGrad<N>
template<int N>
maths::LocalGrad<N> eigenToLocalGrad(
    const Eigen::Matrix<Real, N * 3, 1>& g) {
  maths::LocalGrad<N> result{};
  for (int i = 0; i < N; i++)
    result[i] = glm::dvec3(g(i * 3), g(i * 3 + 1), g(i * 3 + 2));
  return result;
}

} // namespace ksk::fem::ipc::gipc
