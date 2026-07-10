//
// GIPC Mollifier 分支 — 完整的端到端 API
//
// 当两条边近平行时 (crossSquaredNorm < eps_x)，使用 mollifier 软化。
// 本文件提供高层几何 API：输入顶点坐标 + PFPx，输出能量/梯度/Hessian。
// 上层调用者无需知道 I1、I2 等内部不变量。
//
// 数学:
//   能量: E = κ · m_ε(I₁) · ŝ² · (1-I₂)² · [ln(I₂)]²
//   其中 I₁ = ||ea×eb||² (共线度), I₂ = d²/d̂² (归一化距离)
//
// GIPC.cu 约定:
//   - dHat (代码变量) = d̂² (几何距离平方)
//   - _d_EE() 返回 d² (距离平方)
//   - F = diag(1, c, d/d̂) 是对角化形变梯度
//   - PFPx 由 pFpx_pee() 构造 (12×9)
//

#pragma once
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/gipc/hessian.h>
#include <Maths/block-types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <cassert>
#include <cmath>


namespace ksk::fem::ipc::gipc {

// ============================================================================
// 高层结果结构
// ============================================================================

/// Mollifier 分支完整结果 (EE, 4 顶点, 12 DOF)
struct MollifierResult {
  Real energy = 0;
  maths::LocalGrad<4> gradient{};
  maths::LocalHessian<4> hessian{};
  bool active = false;  // 是否在 barrier 激活范围内
};

// ============================================================================
// 高层 API：直接接收顶点坐标，输出完整结果
// ============================================================================

/// 计算 EE mollifier 分支的完整能量 + 梯度 + Hessian
///
/// @param ea0, ea1      第一条边的两个端点 (当前位置)
/// @param eb0, eb1      第二条边的两个端点 (当前位置)
/// @param rest_ea0 ...  4 个 rest-pose 顶点 (用于计算 eps_x)
/// @param dSqr          边-边距离的平方 d² (由外部 _d_EE 计算)
/// @param PFPx          12×9 矩阵 ∂vec(F)/∂x (由 pFpx_pee 计算)
/// @param barrier       barrier 参数 (包含 dHat)
/// @param kappa         刚度系数
///
/// 调用者只需提供几何量和预计算的 PFPx/距离，不需要了解 I1、I2 的定义。
inline MollifierResult computeMollifiedBarrier(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    const glm::dvec3& rest_ea0, const glm::dvec3& rest_ea1,
    const glm::dvec3& rest_eb0, const glm::dvec3& rest_eb1,
    Real dSqr,
    const Eigen::Matrix<Real, 9, 12>& PFPx,
    const Barrier& barrier, Real kappa) {
  MollifierResult result;

  assert(PFPx.allFinite());
  assert(std::isfinite(dSqr));
  assert(std::isfinite(kappa));

  Real dHatSqr = barrier.dHatSqr();
  Real dHat_sqrt = barrier.dHat();  // 几何距离 d̂
  assert(std::isfinite(dHatSqr));
  assert(std::isfinite(dHat_sqrt));
  assert(dHatSqr > 0.0);
  assert(dHat_sqrt > 0.0);

  // === Step 1: I₁ = ||ea × eb||² (共线度) ===
  glm::dvec3 ea = ea1 - ea0;
  glm::dvec3 eb = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(ea, eb);
  Real I1 = glm::dot(crossVec, crossVec);
  assert(std::isfinite(I1));

  if (I1 == 0.0) return result;  // 完全平行 → 能量为 0

  // === Step 2: eps_x (从 rest pose 计算) ===
  Real eps_x = computeEpsCross(rest_ea0, rest_ea1, rest_eb0, rest_eb1);
  assert(std::isfinite(eps_x));

  // === Step 3: I₂ = d²/d̂² ===
  // dSqr 是由外部 _d_EE 计算的真实线段-线段距离平方
  Real I2 = dSqr / dHatSqr;
  assert(std::isfinite(I2));

  if (I2 >= 1.0) return result;  // 不在 barrier 范围内
  if (I2 <= 0.0) return result;

  // === Step 4: 中间量 ===
  Real c = std::sqrt(I1);           // ||ea × eb||
  Real dis = std::sqrt(dSqr);       // 实际距离 d
  Real F33 = dis / dHat_sqrt;       // = √I₂ = d/d̂
  Real m = mollifierValue(I1, eps_x);
  Real L2 = std::log(I2);
  Real t2 = 1.0 - I2;
  assert(std::isfinite(c));
  assert(std::isfinite(dis));
  assert(std::isfinite(F33));
  assert(std::isfinite(m));
  assert(std::isfinite(L2));
  assert(std::isfinite(t2));

  // === Step 5: 能量 ===
  // E = κ · m(I₁) · ŝ² · (1-I₂)² · ln²(I₂)
  Real sHat2 = dHatSqr * dHatSqr;  // = d̂⁴
  assert(std::isfinite(sHat2));
  result.energy = kappa * m * sHat2 * t2 * t2 * L2 * L2;
  assert(std::isfinite(result.energy));
  result.active = true;

  // === Step 6: 对角化 F 和基向量 ===
  // F = diag(1, c, d/d̂) in GIPC coordinate system
  // n1 = (0,1,0): 切向（I₁ 方向）
  // n2 = (0,0,1): 距离方向（I₂ 方向）
  //
  // flatten_g1 = vec(F·n1·n1ᵀ): 只有第 [4] 分量 = c
  // flatten_g2 = vec(F·n2·n2ᵀ): 只有第 [8] 分量 = F33

  // === Step 7: 梯度 ===
  // flatten_pk1 = p1 · flatten_g1 + p2 · flatten_g2
  Real p1 = kappa * barrier.mollifierP1(I1, I2, eps_x);
  Real p2 = kappa * barrier.mollifierP2(I1, I2, eps_x);
  assert(std::isfinite(p1));
  assert(std::isfinite(p2));

  Eigen::Matrix<Real, 9, 1> flatten_pk1 = Eigen::Matrix<Real, 9, 1>::Zero();
  flatten_pk1(4) = p1 * c;       // p1 · flatten_g1[4]
  flatten_pk1(8) = p2 * F33;     // p2 · flatten_g2[8]
  assert(flatten_pk1.allFinite());

  // gradient_12d = PFPx^T · flatten_pk1
  Eigen::Matrix<Real, 12, 1> grad_eigen = PFPx.transpose() * flatten_pk1;
  assert(grad_eigen.allFinite());
  result.gradient = eigenToLocalGrad<4>(grad_eigen);
  assert(std::isfinite(result.gradient[0].x) && std::isfinite(result.gradient[0].y) && std::isfinite(result.gradient[0].z));
  assert(std::isfinite(result.gradient[1].x) && std::isfinite(result.gradient[1].y) && std::isfinite(result.gradient[1].z));
  assert(std::isfinite(result.gradient[2].x) && std::isfinite(result.gradient[2].y) && std::isfinite(result.gradient[2].z));
  assert(std::isfinite(result.gradient[3].x) && std::isfinite(result.gradient[3].y) && std::isfinite(result.gradient[3].z));

  // === Step 8: Hessian (rank ≤ 3) ===
  // 内层 9×9 Hessian 由三组特征对构成:
  //   (1) lambda11, lambda12: twist 方向 q11, q12
  //   (2) (lambda10, lambdag1g, lambda20) 的 2×2 SPD 投影: {q10, q20} 方向
  //
  // 在 GIPC.cu 对角化坐标系下:
  //   q10 = vec(F·n1·n1ᵀ)/√I1 = e₄ (因为 F(1,1)=c, 归一化后=1)
  //   q20 = vec(F·n2·n2ᵀ)/√I2 = e₈ (因为 F(2,2)=F33, 归一化后=1)
  //
  //   q11 = normalize(Tx · vec(F·n1·n1ᵀ))
  //   q12 = normalize(Tz · vec(F·n1·n1ᵀ))
  //   其中 Tx, Tz 是反对称 twist 矩阵

  Real lambda10 = kappa * barrier.mollifierLambda10(I1, I2, eps_x);
  Real lambda11 = kappa * barrier.mollifierLambda11(I1, I2, eps_x);
  Real lambda12 = lambda11;  // λ₁₁ = λ₁₂ (对称性)
  Real lambda20 = kappa * barrier.mollifierLambda20(I1, I2, eps_x);
  Real lambdag1g = kappa * barrier.mollifierLambdaG1G(I1, I2, eps_x);
  assert(std::isfinite(lambda10));
  assert(std::isfinite(lambda11));
  assert(std::isfinite(lambda12));
  assert(std::isfinite(lambda20));
  assert(std::isfinite(lambdag1g));

  Eigen::Matrix<Real, 9, 9> projectedH = Eigen::Matrix<Real, 9, 9>::Zero();

  // --- (1) Twist 方向: q11, q12 ---
  // fnn = F · n1 · n1ᵀ = [[0,0,0],[0,c,0],[0,0,0]] (3×3 矩阵)
  //
  // Tx = (1/√2) * [[0,0,0],[0,0,1],[0,-1,0]]  (GIPC.cu:1897)
  // Tz = (1/√2) * [[0,1,0],[-1,0,0],[0,0,0]]  (GIPC.cu:1899)
  //
  // q11 = normalize(vec(Tx · fnn)):
  //   Tx·fnn: 行0全0; 行1: [0,0,1]·fnn的列→全0(fnn行2全0);
  //           行2: [0,-1,0]·fnn → (2,1)=-c/√2, 其余0
  //   结果矩阵 = [[0,0,0],[0,0,0],[0,-c/√2,0]]
  //   vec(col-major): col0=(0,0,0), col1=(0,0,-c/√2), col2=(0,0,0)
  //   → (0,0,0, 0,0,-c/√2, 0,0,0), 归一化 → ±e₅
  //
  // q12 = normalize(vec(Tz · fnn)):
  //   Tz·fnn: 行0: [0,1,0]·fnn → (0,1)=c/√2, 其余0;
  //           行1: [-1,0,0]·fnn → 全0(fnn行0全0); 行2全0
  //   结果矩阵 = [[0,c/√2,0],[0,0,0],[0,0,0]]
  //   vec(col-major): col0=(0,0,0), col1=(c/√2,0,0), col2=(0,0,0)
  //   → (0,0,0, c/√2,0,0, 0,0,0), 归一化 → ±e₃
  //
  // 外积 q·qᵀ 中符号不影响

  // q11 → e₅
  if (lambda11 > 0) {
    projectedH(5, 5) += lambda11;
  }
  // q12 → e₃
  if (lambda12 > 0) {
    projectedH(3, 3) += lambda12;
  }

  // --- (2) 距离-切向耦合 2×2 子问题 ---
  // 在 {q10=e₄, q20=e₈} 子空间中:
  //   | lambda10   lambdag1g |
  //   | lambdag1g  lambda20  |
  // 做 SPD 投影后重构到 9×9 空间

  auto pd = makePD2x2(lambda10, lambdag1g, lambda20);
  for (int i = 0; i < pd.numPositive; i++) {
    // 特征向量在 {e₄, e₈} 子空间: (a, b) → 9维: a*e₄ + b*e₈
    Real a = pd.eigenVecs[i](0);
    Real b = pd.eigenVecs[i](1);
    Real lam = pd.eigenValues[i];
    projectedH(4, 4) += lam * a * a;
    projectedH(4, 8) += lam * a * b;
    projectedH(8, 4) += lam * b * a;
    projectedH(8, 8) += lam * b * b;
  }

  assert(projectedH.allFinite());

  // --- Sandwich: H_12x12 = PFPx^T · projectedH · PFPx ---
  result.hessian = sandwichFull<4, 9>(PFPx, projectedH);

  return result;
}


/// 简化版: 只计算能量（不需要 PFPx，用于能量求值/line search）
///
/// @param ea0, ea1, eb0, eb1  当前位置顶点
/// @param rest_ea0 ...        rest-pose 顶点
/// @param dSqr                边-边距离平方 (由 _d_EE 计算)
/// @param barrier             barrier 参数
/// @param kappa               刚度系数
inline Real computeMollifiedBarrierEnergy(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    const glm::dvec3& rest_ea0, const glm::dvec3& rest_ea1,
    const glm::dvec3& rest_eb0, const glm::dvec3& rest_eb1,
    Real dSqr,
    const Barrier& barrier, Real kappa) {
  Real dHatSqr = barrier.dHatSqr();

  // I₁
  glm::dvec3 ea = ea1 - ea0;
  glm::dvec3 eb = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(ea, eb);
  Real I1 = glm::dot(crossVec, crossVec);
  if (I1 == 0.0) return 0.0;

  // I₂
  Real I2 = dSqr / dHatSqr;
  if (I2 >= 1.0 || I2 <= 0.0) return 0.0;

  // eps_x
  Real eps_x = computeEpsCross(rest_ea0, rest_ea1, rest_eb0, rest_eb1);

  // E = κ · m(I₁) · ŝ² · (1-I₂)² · ln²(I₂)
  Real m = mollifierValue(I1, eps_x);
  Real L2 = std::log(I2);
  Real t2 = 1.0 - I2;
  Real sHat2 = dHatSqr * dHatSqr;
  return kappa * m * sHat2 * t2 * t2 * L2 * L2;
}

/// 判断一对 EE 是否需要走 mollifier 分支
///
/// @param ea0, ea1, eb0, eb1           当前位置顶点
/// @param rest_ea0, rest_ea1, ...      rest-pose 顶点
/// @return true 表示 I₁ < eps_x，应走 mollifier 分支
inline bool needsMollifier(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    const glm::dvec3& rest_ea0, const glm::dvec3& rest_ea1,
    const glm::dvec3& rest_eb0, const glm::dvec3& rest_eb1) {
  glm::dvec3 ea = ea1 - ea0;
  glm::dvec3 eb = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(ea, eb);
  Real I1 = glm::dot(crossVec, crossVec);
  Real eps_x = computeEpsCross(rest_ea0, rest_ea1, rest_eb0, rest_eb1);
  return I1 < eps_x;
}

// ============================================================================
// 内部诊断: 9×9 内层 Hessian (用于测试验证)
// 构造 mollifier 分支在对角化 vec(F) 空间中的 9×9 SPD Hessian
// 这是一个内部函数，正常代码路径直接调用 computeMollifiedBarrier。
// ============================================================================

/// 构造 mollifier 分支的 9×9 内层 Hessian (SPD 投影后)
/// @param I1     ||ea × eb||² (共线度)
/// @param I2     d²/d̂² (归一化距离)
/// @param eps_x  rest-pose mollifier 阈值
/// @param barrier  barrier 参数
/// @param kappa    刚度系数
inline Eigen::Matrix<Real, 9, 9> computeMollifierInnerHessian(
    Real I1, Real I2, Real eps_x,
    const Barrier& barrier, Real kappa) {
  Eigen::Matrix<Real, 9, 9> H = Eigen::Matrix<Real, 9, 9>::Zero();

  if (I1 <= 0 || I2 >= 1.0 || I2 <= 0) return H;

  // 特征值系数
  Real lambda10 = kappa * barrier.mollifierLambda10(I1, I2, eps_x);
  Real lambda11 = kappa * barrier.mollifierLambda11(I1, I2, eps_x);
  Real lambda12 = lambda11;  // 对称性
  Real lambda20 = kappa * barrier.mollifierLambda20(I1, I2, eps_x);
  Real lambdag1g = kappa * barrier.mollifierLambdaG1G(I1, I2, eps_x);

  // === Twist 方向 (lambda11, lambda12) ===
  // 在 GIPC.cu 对角化坐标系下:
  //   fnn = F · n1 · n1ᵀ = [[0,0,0],[0,c,0],[0,0,0]]
  //   q11 = normalize(vec(Tx · fnn)):
  //     Tx·fnn = [[0,0,0],[0,0,0],[0,-c/√2,0]]
  //     vec(col-major) = (0,0,0, 0,0,-c/√2, 0,0,0), 归一化 → ±e₅
  //   q12 = normalize(vec(Tz · fnn)):
  //     Tz·fnn = [[0,c/√2,0],[0,0,0],[0,0,0]]
  //     vec(col-major) = (0,0,0, c/√2,0,0, 0,0,0), 归一化 → ±e₃
  if (lambda11 > 0) {
    H(5, 5) += lambda11;
  }
  if (lambda12 > 0) {
    H(3, 3) += lambda12;
  }

  // === 距离-切向耦合 2×2 子问题 ===
  // {q10=e₄, q20=e₈} 子空间
  auto pd = makePD2x2(lambda10, lambdag1g, lambda20);
  for (int i = 0; i < pd.numPositive; i++) {
    Real a = pd.eigenVecs[i](0);
    Real b = pd.eigenVecs[i](1);
    Real lam = pd.eigenValues[i];
    H(4, 4) += lam * a * a;
    H(4, 8) += lam * a * b;
    H(8, 4) += lam * b * a;
    H(8, 8) += lam * b * b;
  }

  return H;
}

} // namespace ksk::fem::ipc::gipc
