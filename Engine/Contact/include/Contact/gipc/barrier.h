//
// GIPC Barrier (RANK=2)
//
// b鈧?I鈧? = 魏 路 艥虏 路 (I鈧?1)虏 路 [ln(I鈧?]虏
// I鈧?= d虏/d虃虏, 艥 = d虃虏
//
//   - dHat is d虃
//   - dHatSqr() = d虃虏 = GIPC.cu "dHat"
//   - all _d_* returns d虏
//   - I鈧?= d虏/d虃虏
//

#pragma once
#include <Contact/types.h>
#include <cmath>
#include <algorithm>

namespace ksk::engine::contact::gipc {

struct Barrier {
  explicit Barrier(Real dHat) : m_dHat(dHat) {}

  [[nodiscard]] Real dHat() const { return m_dHat; }
  [[nodiscard]] Real dHatSqr() const { return m_dHat * m_dHat; }

  [[nodiscard]] Real energy(Real dSqr) const {
    if (dSqr >= dHatSqr()) return 0.0;
    if (dSqr <= 0.0) return 1e18;
    Real I5 = dSqr / dHatSqr();
    Real L = std::log(I5);
    Real t = dSqr - dHatSqr();
    return t * t * L * L;
  }

  /// GIPC.cu flatten_pk1 = tmp * gradCoeff(I5)
  [[nodiscard]] Real gradCoeff(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return 4.0 * sHat2() * L * (I5 - 1.0) * (I5 + I5 * L - 1.0) / I5;
  }

  [[nodiscard]] Real dBdI5(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return 2.0 * sHat2() * L * (I5 - 1.0) * (I5 + I5 * L - 1.0) / I5;
  }

  // =========================================================================
  // Hessian
  // =========================================================================

  ///  (RANK=2)
  /// 鏉ヨ嚜 GIPC.cu:1757-1761 (鍘绘帀 Kappa 鍓嶇紑)
  [[nodiscard]] Real lambda0(Real I5) const {
    if (I5 >= 1.0 || I5 <= 0.0) return 0.0;
    Real L = std::log(I5);
    return -(4.0 * sHat2()
             * (4.0 * I5 + L - 3.0 * I5 * I5 * L * L + 6.0 * I5 * L
                - 2.0 * I5 * I5 + I5 * L * L - 7.0 * I5 * I5 * L - 2.0))
           / I5;
  }

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
  // Mollifier 鍒嗘敮鐨勬爣閲忕郴鏁?(RANK=2) 鈥?涓嶅惈 魏
  //
  // 杩欎簺鏄唴閮ㄥ疄鐜扮粏鑺傦紝涓婂眰搴旈€氳繃 mollifier.h 涓殑楂樺眰 API 浣跨敤銆?
  // 鍙傛暟璇存槑:
  //   I1    = ||ea 脳 eb||虏      (鍏辩嚎搴﹀钩鏂?
  //   I2    = d虏/d虃虏             (褰掍竴鍖栬窛绂诲钩鏂?= d_EE()虏 / dHatSqr)
  //   eps_x = computeEpsCross() (rest pose 闃堝€?
  // =========================================================================

  /// Mollifier 姊害绯绘暟 p鈧?(瀵?I鈧?姹傚椤? 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: p1 = -Kappa * 2 * (2*dHat虏*ln虏(I2)*(I1-eps_x)*(I2-1)虏) / eps_x虏
  [[nodiscard]] Real mollifierP1(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -4.0 * sHat2() * L2 * L2 * (I1 - eps_x)
           * (I2 - 1.0) * (I2 - 1.0)
           / (eps_x * eps_x);
  }

  /// Mollifier 姊害绯绘暟 p鈧?(瀵?I鈧?姹傚椤? 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: p2 = -Kappa * 2 * (2*I1*dHat虏*ln(I2)*(I1-2eps_x)*(I2-1)*(I2+I2*ln(I2)-1)) / (I2*eps_x虏)
  [[nodiscard]] Real mollifierP2(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    Real L2 = std::log(I2);
    return -4.0 * I1 * sHat2() * L2 * (I1 - 2.0 * eps_x)
           * (I2 - 1.0) * (I2 + I2 * L2 - 1.0)
           / (I2 * eps_x * eps_x);
  }

  /// Mollifier Hessian: 位鈧佲個 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: lambda10 = -Kappa * (4*dHat虏*ln虏(I2)*(I2-1)虏*(3*I1-eps_x)) / eps_x虏
  [[nodiscard]] Real mollifierLambda10(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -(4.0 * sHat2() * L2 * L2 * (I2 - 1.0) * (I2 - 1.0)
             * (3.0 * I1 - eps_x))
           / (eps_x * eps_x);
  }

  /// Mollifier Hessian: 位鈧佲倎 = 位鈧佲倐 (twist 鏂瑰悜鏇茬巼) 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: lambda11 = -Kappa * (4*dHat虏*ln虏(I2)*(I1-eps_x)*(I2-1)虏) / eps_x虏
  [[nodiscard]] Real mollifierLambda11(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0) return 0.0;
    Real L2 = std::log(I2);
    return -(4.0 * sHat2() * L2 * L2 * (I1 - eps_x)
             * (I2 - 1.0) * (I2 - 1.0))
           / (eps_x * eps_x);
  }

  /// Mollifier Hessian: 位鈧傗個 (璺濈鏂瑰悜浜岄樁锛屽惈 mollifier 鏉冮噸) 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: lambda20 = +Kappa * (4*I1*dHat虏*(I1-2*eps_x)*[澶氶」寮廬) / (I2*eps_x虏)
  /// 娉ㄦ剰: 绗﹀彿涓烘锛堜笌 lambda10 鐩稿弽锛?
  [[nodiscard]] Real mollifierLambda20(Real I1, Real I2, Real eps_x) const {
    if (I2 >= 1.0 || I2 <= 0.0) return 0.0;
    Real L2 = std::log(I2);
    return (4.0 * sHat2() * I1 * (I1 - 2.0 * eps_x)
            * (4.0 * I2 + L2 - 3.0 * I2 * I2 * L2 * L2 + 6.0 * I2 * L2
               - 2.0 * I2 * I2 + I2 * L2 * L2 - 7.0 * I2 * I2 * L2 - 2.0))
           / (I2 * eps_x * eps_x);
  }

  /// Mollifier Hessian: 位_g1g (I鈧?I鈧?浜ゅ弶椤? 鈥?涓嶅惈 魏
  /// GIPC.cu RANK=2: lambdag1g = -Kappa * 4*c*F33 * (4*dHat虏*ln(I2)*(I1-eps_x)*(I2-1)*(I2+I2*ln(I2)-1)) / (I2*eps_x虏)
  /// 鍏朵腑 c = 鈭欼鈧? F33 = 鈭欼鈧?
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
// Mollifier 鏍囬噺鍑芥暟 (鐙珛浜?Barrier 缁撴瀯浣?
// ============================================================================

/// Mollifier 鍑芥暟 m_蔚(c) = 2c/蔚 - c虏/蔚虏  (c < 蔚)锛屽惁鍒?1
/// c = I鈧?= ||ea 脳 eb||虏
inline Real mollifierValue(Real c, Real eps_x) {
  if (c >= eps_x) return 1.0;
  return (2.0 / eps_x) * c - (1.0 / (eps_x * eps_x)) * c * c;
}

/// Mollifier 涓€闃跺 m'(c) = 2/蔚 - 2c/蔚虏  (c < 蔚)锛屽惁鍒?0
inline Real mollifierDerivative(Real c, Real eps_x) {
  if (c >= eps_x) return 0.0;
  return 2.0 / eps_x - 2.0 * c / (eps_x * eps_x);
}

/// Mollifier 浜岄樁瀵?m''(c) = -2/蔚虏  (c < 蔚)锛屽惁鍒?0
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

} // namespace ksk::engine::contact::gipc
