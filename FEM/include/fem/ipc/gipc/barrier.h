//
// GIPC Barrier (RANK=2) — 标量 barrier 函数 + mollifier 标量系数
//
// 能量: b₂(I₅) = κ · ŝ² · (I₅-1)² · [ln(I₅)]²
// 其中 I₅ = d²/d̂², ŝ = d̂²
//
// 约定（与 GIPC.cu 一致）:
//   - 构造参数 dHat 是几何距离阈值 d̂
//   - dHatSqr() = d̂² = GIPC.cu 中变量名 "dHat" 的值
//   - 所有 _d_* 距离函数返回 d²
//   - I₅ = d²/d̂²，barrier 在 I₅ ∈ (0, 1) 时激活
//
// 比经典 IPC 的 RANK=1 barrier 有更高光滑度 (C³ at d=d̂)
// 和更好的 Newton 收敛性
//

#pragma once
#include <fem/types.h>
#include <cmath>
#include <algorithm>

namespace ksk::fem::ipc::gipc {

struct Barrier {
  explicit Barrier(Real dHat) : m_dHat(dHat) {}

  /// 距离阈值 d̂ (几何距离)
  [[nodiscard]] Real dHat() const { return m_dHat; }
  /// d̂² — 这是 GIPC.cu 中变量 "dHat" 的真实含义
  [[nodiscard]] Real dHatSqr() const { return m_dHat * m_dHat; }

  // =========================================================================
  // 能量
  // =========================================================================

  /// barrier 能量 b(d²)，输入距离平方 dSqr = d²
  /// 返回 ŝ² · (I₅-1)² · ln²(I₅)  (不含 κ，由外部乘)
  [[nodiscard]] Real energy(Real dSqr) const {
    if (dSqr >= dHatSqr()) return 0.0;
    if (dSqr <= 0.0) return 1e18;  // 穿透保护
    Real I5 = dSqr / dHatSqr();
    Real L = std::log(I5);
    Real t = dSqr - dHatSqr();
    return t * t * L * L;
  }

  // =========================================================================
  // 梯度所需：flatten_pk1 的标量系数
  // =========================================================================

  /// 梯度完整标量系数 = 2 · ∂b/∂I₅ (已含链式法则因子 2)
  /// GIPC.cu 中: flatten_pk1 = tmp * gradCoeff(I5)
  [[nodiscard]] Real gradCoeff(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return 4.0 * sHat2() * L * (I5 - 1.0) * (I5 + I5 * L - 1.0) / I5;
  }

  /// 纯数学 ∂b/∂I₅ (不含链式法则因子2，用于验证)
  [[nodiscard]] Real dBdI5(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return 2.0 * sHat2() * L * (I5 - 1.0) * (I5 + I5 * L - 1.0) / I5;
  }

  // =========================================================================
  // Hessian 所需：λ₀ (简单分支内层 9×9 Hessian 的唯一非零特征值)
  // =========================================================================

  /// λ₀ (RANK=2): 解析闭合公式 — 不含 κ
  /// 来自 GIPC.cu:1757-1761 (去掉 Kappa 前缀)
  [[nodiscard]] Real lambda0(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return -(4.0 * sHat2()
             * (4.0 * I5 + L - 3.0 * I5 * I5 * L * L + 6.0 * I5 * L
                - 2.0 * I5 * I5 + I5 * L * L - 7.0 * I5 * I5 * L - 2.0))
           / I5;
  }

  /// Gauss 阈值保护后的 λ₀ (SPD 投影)
  [[nodiscard]] Real clampedLambda0(Real I5) const {
    if (I5 >= 1.0) return 0.0;
    Real lam;
    if (I5 < GAUSS_THRESHOLD) {
      lam = lambda0(GAUSS_THRESHOLD);
    } else {
      lam = lambda0(I5);
    }
    return std::max(lam, 0.0);
  }

  // =========================================================================
  // Mollifier 分支的标量系数 (RANK=2) — 不含 κ
  //
  // 这些是内部实现细节，上层应通过 mollifier.h 中的高层 API 使用。
  // 参数说明:
  //   I1    = ||ea × eb||²      (共线度平方)
  //   I2    = d²/d̂²             (归一化距离平方 = d_EE()² / dHatSqr)
  //   eps_x = computeEpsCross() (rest pose 阈值)
  // =========================================================================

  /// Mollifier 梯度系数 p₁ (对 I₁ 求导项) — 不含 κ
  /// GIPC.cu RANK=2: p1 = -Kappa * 2 * (2*dHat²*ln²(I2)*(I1-eps_x)*(I2-1)²) / eps_x²
  [[nodiscard]] Real mollifierP1(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -4.0 * sHat2() * L2 * L2 * (I1 - eps_x)
           * (I2 - 1.0) * (I2 - 1.0)
           / (eps_x * eps_x);
  }

  /// Mollifier 梯度系数 p₂ (对 I₂ 求导项) — 不含 κ
  /// GIPC.cu RANK=2: p2 = -Kappa * 2 * (2*I1*dHat²*ln(I2)*(I1-2eps_x)*(I2-1)*(I2+I2*ln(I2)-1)) / (I2*eps_x²)
  [[nodiscard]] Real mollifierP2(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    Real L2 = std::log(I2);
    return -4.0 * I1 * sHat2() * L2 * (I1 - 2.0 * eps_x)
           * (I2 - 1.0) * (I2 + I2 * L2 - 1.0)
           / (I2 * eps_x * eps_x);
  }

  /// Mollifier Hessian: λ₁₀ — 不含 κ
  /// GIPC.cu RANK=2: lambda10 = -Kappa * (4*dHat²*ln²(I2)*(I2-1)²*(3*I1-eps_x)) / eps_x²
  [[nodiscard]] Real mollifierLambda10(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -(4.0 * sHat2() * L2 * L2 * (I2 - 1.0) * (I2 - 1.0)
             * (3.0 * I1 - eps_x))
           / (eps_x * eps_x);
  }

  /// Mollifier Hessian: λ₁₁ = λ₁₂ (twist 方向曲率) — 不含 κ
  /// GIPC.cu RANK=2: lambda11 = -Kappa * (4*dHat²*ln²(I2)*(I1-eps_x)*(I2-1)²) / eps_x²
  [[nodiscard]] Real mollifierLambda11(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -(4.0 * sHat2() * L2 * L2 * (I1 - eps_x)
             * (I2 - 1.0) * (I2 - 1.0))
           / (eps_x * eps_x);
  }

  /// Mollifier Hessian: λ₂₀ (距离方向二阶，含 mollifier 权重) — 不含 κ
  /// GIPC.cu RANK=2: lambda20 = +Kappa * (4*I1*dHat²*(I1-2*eps_x)*[多项式]) / (I2*eps_x²)
  /// 注意: 符号为正（与 lambda10 相反）
  [[nodiscard]] Real mollifierLambda20(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    Real L2 = std::log(I2);
    return (4.0 * sHat2() * I1 * (I1 - 2.0 * eps_x)
            * (4.0 * I2 + L2 - 3.0 * I2 * I2 * L2 * L2 + 6.0 * I2 * L2
               - 2.0 * I2 * I2 + I2 * L2 * L2 - 7.0 * I2 * I2 * L2 - 2.0))
           / (I2 * eps_x * eps_x);
  }

  /// Mollifier Hessian: λ_g1g (I₁-I₂ 交叉项) — 不含 κ
  /// GIPC.cu RANK=2: lambdag1g = -Kappa * 4*c*F33 * (4*dHat²*ln(I2)*(I1-eps_x)*(I2-1)*(I2+I2*ln(I2)-1)) / (I2*eps_x²)
  /// 其中 c = √I₁, F33 = √I₂
  [[nodiscard]] Real mollifierLambdaG1G(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0 || I2 <= 0.0 || I1 <= 0.0) return 0.0;
    Real L2 = std::log(I2);
    Real c = std::sqrt(I1);
    Real F33 = std::sqrt(I2);
    return -4.0 * c * F33 * sHat2() * L2 * (I1 - eps_x)
           * (I2 - 1.0) * (I2 + I2 * L2 - 1.0)
           / (I2 * eps_x * eps_x);
  }

  static constexpr Real GAUSS_THRESHOLD = 1e-4;

private:
  Real m_dHat;

  [[nodiscard]] Real sHat2() const { return dHatSqr() * dHatSqr(); }
};

// ============================================================================
// Mollifier 标量函数 (独立于 Barrier 结构体)
// ============================================================================

/// Mollifier 函数 m_ε(c) = 2c/ε - c²/ε²  (c < ε)，否则 1
/// c = I₁ = ||ea × eb||²
inline Real mollifierValue(Real c, Real eps_x) {
  if (c >= eps_x) return 1.0;
  return (2.0 / eps_x) * c - (1.0 / (eps_x * eps_x)) * c * c;
}

/// Mollifier 一阶导 m'(c) = 2/ε - 2c/ε²  (c < ε)，否则 0
inline Real mollifierDerivative(Real c, Real eps_x) {
  if (c >= eps_x) return 0.0;
  return 2.0 / eps_x - 2.0 * c / (eps_x * eps_x);
}

/// Mollifier 二阶导 m''(c) = -2/ε²  (c < ε)，否则 0
inline Real mollifierSecondDerivative(Real c, Real eps_x) {
  if (c >= eps_x) return 0.0;
  return -2.0 / (eps_x * eps_x);
}

inline Real computeEpsCross(const glm::dvec3& ea0, const glm::dvec3& ea1,
                            const glm::dvec3& eb0, const glm::dvec3& eb1) {
  Real la2 = glm::dot(ea1 - ea0, ea1 - ea0);
  Real lb2 = glm::dot(eb1 - eb0, eb1 - eb0);
  return 1e-3 * la2 * lb2;
}

} // namespace ksk::fem::ipc::gipc
