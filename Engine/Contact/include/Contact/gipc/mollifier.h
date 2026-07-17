//
// GIPC Mollifier 鍒嗘敮 鈥?瀹屾暣鐨勭鍒扮 API
//
// 褰撲袱鏉¤竟杩戝钩琛屾椂 (crossSquaredNorm < eps_x)锛屼娇鐢?mollifier 杞寲銆?
// 鏈枃浠舵彁渚涢珮灞傚嚑浣?API锛氳緭鍏ラ《鐐瑰潗鏍?+ PFPx锛岃緭鍑鸿兘閲?姊害/Hessian銆?
// 涓婂眰璋冪敤鑰呮棤闇€鐭ラ亾 I1銆両2 绛夊唴閮ㄤ笉鍙橀噺銆?
//
// 鏁板:
//   鑳介噺: E = 魏 路 m_蔚(I鈧? 路 艥虏 路 (1-I鈧?虏 路 [ln(I鈧?]虏
//   鍏朵腑 I鈧?= ||ea脳eb||虏 (鍏辩嚎搴?, I鈧?= d虏/d虃虏 (褰掍竴鍖栬窛绂?
//
// GIPC.cu 绾﹀畾:
//   - dHat (浠ｇ爜鍙橀噺) = d虃虏 (鍑犱綍璺濈骞虫柟)
//   - _d_EE() 杩斿洖 d虏 (璺濈骞虫柟)
//   - F = diag(1, c, d/d虃) 鏄瑙掑寲褰㈠彉姊害
//   - PFPx 鐢?pFpx_pee() 鏋勯€?(12脳9)
//

#pragma once
#include <Contact/gipc/barrier.h>
#include <Contact/gipc/hessian.h>
#include <Maths/block-types.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <cassert>
#include <cmath>


namespace ksk::engine::contact::gipc {

// ============================================================================
// 楂樺眰缁撴灉缁撴瀯
// ============================================================================

struct MollifierResult {
  Real energy = 0;
  maths::LocalGrad<4> gradient{};
  maths::LocalHessian<4> hessian{};
  bool active = false;  // 鏄惁鍦?barrier 婵€娲昏寖鍥村唴
};

// ============================================================================
// 楂樺眰 API锛氱洿鎺ユ帴鏀堕《鐐瑰潗鏍囷紝杈撳嚭瀹屾暣缁撴灉
// ============================================================================

/// 璁＄畻 EE mollifier 鍒嗘敮鐨勫畬鏁磋兘閲?+ 姊害 + Hessian
///
/// @param ea0, ea1      绗竴鏉¤竟鐨勪袱涓鐐?(褰撳墠浣嶇疆)
/// @param eb0, eb1      绗簩鏉¤竟鐨勪袱涓鐐?(褰撳墠浣嶇疆)
/// @param rest_ea0 ...  4 涓?rest-pose 椤剁偣 (鐢ㄤ簬璁＄畻 eps_x)
/// @param dSqr          杈?杈硅窛绂荤殑骞虫柟 d虏 (鐢卞閮?_d_EE 璁＄畻)
/// @param PFPx          12脳9 鐭╅樀 鈭倂ec(F)/鈭倄 (鐢?pFpx_pee 璁＄畻)
/// @param barrier       barrier 鍙傛暟 (鍖呭惈 dHat)
/// @param kappa         鍒氬害绯绘暟
///
/// 璋冪敤鑰呭彧闇€鎻愪緵鍑犱綍閲忓拰棰勮绠楃殑 PFPx/璺濈锛屼笉闇€瑕佷簡瑙?I1銆両2 鐨勫畾涔夈€?
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
  Real dHat_sqrt = barrier.dHat();  // 鍑犱綍璺濈 d虃
  assert(std::isfinite(dHatSqr));
  assert(std::isfinite(dHat_sqrt));
  assert(dHatSqr > 0.0);
  assert(dHat_sqrt > 0.0);

  // === Step 1: I鈧?= ||ea 脳 eb||虏 (鍏辩嚎搴? ===
  glm::dvec3 ea = ea1 - ea0;
  glm::dvec3 eb = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(ea, eb);
  Real I1 = glm::dot(crossVec, crossVec);
  assert(std::isfinite(I1));

  if (I1 == 0.0) return result;  // 瀹屽叏骞宠 鈫?鑳介噺涓?0

  // === Step 2: eps_x (浠?rest pose 璁＄畻) ===
  Real eps_x = computeEpsCross(rest_ea0, rest_ea1, rest_eb0, rest_eb1);
  assert(std::isfinite(eps_x));

  // === Step 3: I鈧?= d虏/d虃虏 ===
  // dSqr 鏄敱澶栭儴 _d_EE 璁＄畻鐨勭湡瀹炵嚎娈?绾挎璺濈骞虫柟
  Real I2 = dSqr / dHatSqr;
  assert(std::isfinite(I2));

  if (I2 >= 1.0) return result;  // 涓嶅湪 barrier 鑼冨洿鍐?
  if (I2 <= 0.0) return result;

  // === Step 4: 涓棿閲?===
  Real c = std::sqrt(I1);           // ||ea 脳 eb||
  Real dis = std::sqrt(dSqr);       // 瀹為檯璺濈 d
  Real F33 = dis / dHat_sqrt;       // = 鈭欼鈧?= d/d虃
  Real m = mollifierValue(I1, eps_x);
  Real L2 = std::log(I2);
  Real t2 = 1.0 - I2;
  assert(std::isfinite(c));
  assert(std::isfinite(dis));
  assert(std::isfinite(F33));
  assert(std::isfinite(m));
  assert(std::isfinite(L2));
  assert(std::isfinite(t2));

  // === Step 5: 鑳介噺 ===
  // E = 魏 路 m(I鈧? 路 艥虏 路 (1-I鈧?虏 路 ln虏(I鈧?
  Real sHat2 = dHatSqr * dHatSqr;  // = d虃鈦?
  assert(std::isfinite(sHat2));
  result.energy = kappa * m * sHat2 * t2 * t2 * L2 * L2;
  assert(std::isfinite(result.energy));
  result.active = true;

  // === Step 6: 瀵硅鍖?F 鍜屽熀鍚戦噺 ===
  // F = diag(1, c, d/d虃) in GIPC coordinate system
  // n1 = (0,1,0): 鍒囧悜锛圛鈧?鏂瑰悜锛?
  // n2 = (0,0,1): 璺濈鏂瑰悜锛圛鈧?鏂瑰悜锛?
  //
  // flatten_g1 = vec(F路n1路n1岬€): 鍙湁绗?[4] 鍒嗛噺 = c
  // flatten_g2 = vec(F路n2路n2岬€): 鍙湁绗?[8] 鍒嗛噺 = F33

  // === Step 7: 姊害 ===
  // flatten_pk1 = p1 路 flatten_g1 + p2 路 flatten_g2
  Real p1 = kappa * barrier.mollifierP1(I1, I2, eps_x);
  Real p2 = kappa * barrier.mollifierP2(I1, I2, eps_x);
  assert(std::isfinite(p1));
  assert(std::isfinite(p2));

  Eigen::Matrix<Real, 9, 1> flatten_pk1 = Eigen::Matrix<Real, 9, 1>::Zero();
  flatten_pk1(4) = p1 * c;       // p1 路 flatten_g1[4]
  flatten_pk1(8) = p2 * F33;     // p2 路 flatten_g2[8]
  assert(flatten_pk1.allFinite());

  // gradient_12d = PFPx^T 路 flatten_pk1
  Eigen::Matrix<Real, 12, 1> grad_eigen = PFPx.transpose() * flatten_pk1;
  assert(grad_eigen.allFinite());
  result.gradient = eigenToLocalGrad<4>(grad_eigen);
  assert(std::isfinite(result.gradient[0].x) && std::isfinite(result.gradient[0].y) && std::isfinite(result.gradient[0].z));
  assert(std::isfinite(result.gradient[1].x) && std::isfinite(result.gradient[1].y) && std::isfinite(result.gradient[1].z));
  assert(std::isfinite(result.gradient[2].x) && std::isfinite(result.gradient[2].y) && std::isfinite(result.gradient[2].z));
  assert(std::isfinite(result.gradient[3].x) && std::isfinite(result.gradient[3].y) && std::isfinite(result.gradient[3].z));

  // === Step 8: Hessian (rank 鈮?3) ===
  // 鍐呭眰 9脳9 Hessian 鐢变笁缁勭壒寰佸鏋勬垚:
  //   (1) lambda11, lambda12: twist 鏂瑰悜 q11, q12
  //   (2) (lambda10, lambdag1g, lambda20) 鐨?2脳2 SPD 鎶曞奖: {q10, q20} 鏂瑰悜
  //
  // 鍦?GIPC.cu 瀵硅鍖栧潗鏍囩郴涓?
  //   q10 = vec(F路n1路n1岬€)/鈭欼1 = e鈧?(鍥犱负 F(1,1)=c, 褰掍竴鍖栧悗=1)
  //   q20 = vec(F路n2路n2岬€)/鈭欼2 = e鈧?(鍥犱负 F(2,2)=F33, 褰掍竴鍖栧悗=1)
  //
  //   q11 = normalize(Tx 路 vec(F路n1路n1岬€))
  //   q12 = normalize(Tz 路 vec(F路n1路n1岬€))
  //   鍏朵腑 Tx, Tz 鏄弽瀵圭О twist 鐭╅樀

  Real lambda10 = kappa * barrier.mollifierLambda10(I1, I2, eps_x);
  Real lambda11 = kappa * barrier.mollifierLambda11(I1, I2, eps_x);
  Real lambda12 = lambda11;  // 位鈧佲倎 = 位鈧佲倐 (瀵圭О鎬?
  Real lambda20 = kappa * barrier.mollifierLambda20(I1, I2, eps_x);
  Real lambdag1g = kappa * barrier.mollifierLambdaG1G(I1, I2, eps_x);
  assert(std::isfinite(lambda10));
  assert(std::isfinite(lambda11));
  assert(std::isfinite(lambda12));
  assert(std::isfinite(lambda20));
  assert(std::isfinite(lambdag1g));

  Eigen::Matrix<Real, 9, 9> projectedH = Eigen::Matrix<Real, 9, 9>::Zero();

  // --- (1) Twist 鏂瑰悜: q11, q12 ---
  // fnn = F 路 n1 路 n1岬€ = [[0,0,0],[0,c,0],[0,0,0]] (3脳3 鐭╅樀)
  //
  // Tx = (1/鈭?) * [[0,0,0],[0,0,1],[0,-1,0]]  (GIPC.cu:1897)
  // Tz = (1/鈭?) * [[0,1,0],[-1,0,0],[0,0,0]]  (GIPC.cu:1899)
  //
  // q11 = normalize(vec(Tx 路 fnn)):
  //   Tx路fnn: 琛?鍏?; 琛?: [0,0,1]路fnn鐨勫垪鈫掑叏0(fnn琛?鍏?);
  //           琛?: [0,-1,0]路fnn 鈫?(2,1)=-c/鈭?, 鍏朵綑0
  //   缁撴灉鐭╅樀 = [[0,0,0],[0,0,0],[0,-c/鈭?,0]]
  //   vec(col-major): col0=(0,0,0), col1=(0,0,-c/鈭?), col2=(0,0,0)
  //   鈫?(0,0,0, 0,0,-c/鈭?, 0,0,0), 褰掍竴鍖?鈫?卤e鈧?
  //
  // q12 = normalize(vec(Tz 路 fnn)):
  //   Tz路fnn: 琛?: [0,1,0]路fnn 鈫?(0,1)=c/鈭?, 鍏朵綑0;
  //           琛?: [-1,0,0]路fnn 鈫?鍏?(fnn琛?鍏?); 琛?鍏?
  //   缁撴灉鐭╅樀 = [[0,c/鈭?,0],[0,0,0],[0,0,0]]
  //   vec(col-major): col0=(0,0,0), col1=(c/鈭?,0,0), col2=(0,0,0)
  //   鈫?(0,0,0, c/鈭?,0,0, 0,0,0), 褰掍竴鍖?鈫?卤e鈧?
  //
  // 澶栫Н q路q岬€ 涓鍙蜂笉褰卞搷

  // q11 鈫?e鈧?
  if (lambda11 > 0) {
    projectedH(5, 5) += lambda11;
  }
  // q12 鈫?e鈧?
  if (lambda12 > 0) {
    projectedH(3, 3) += lambda12;
  }

  // --- (2) 璺濈-鍒囧悜鑰﹀悎 2脳2 瀛愰棶棰?---
  // 鍦?{q10=e鈧? q20=e鈧坿 瀛愮┖闂翠腑:
  //   | lambda10   lambdag1g |
  //   | lambdag1g  lambda20  |
  // 鍋?SPD 鎶曞奖鍚庨噸鏋勫埌 9脳9 绌洪棿

  auto pd = makePD2x2(lambda10, lambdag1g, lambda20);
  for (int i = 0; i < pd.numPositive; i++) {
    // 鐗瑰緛鍚戦噺鍦?{e鈧? e鈧坿 瀛愮┖闂? (a, b) 鈫?9缁? a*e鈧?+ b*e鈧?
    Real a = pd.eigenVecs[i](0);
    Real b = pd.eigenVecs[i](1);
    Real lam = pd.eigenValues[i];
    projectedH(4, 4) += lam * a * a;
    projectedH(4, 8) += lam * a * b;
    projectedH(8, 4) += lam * b * a;
    projectedH(8, 8) += lam * b * b;
  }

  assert(projectedH.allFinite());

  // --- Sandwich: H_12x12 = PFPx^T 路 projectedH 路 PFPx ---
  result.hessian = sandwichFull<4, 9>(PFPx, projectedH);

  return result;
}


/// 绠€鍖栫増: 鍙绠楄兘閲忥紙涓嶉渶瑕?PFPx锛岀敤浜庤兘閲忔眰鍊?line search锛?
///
/// @param ea0, ea1, eb0, eb1  褰撳墠浣嶇疆椤剁偣
/// @param rest_ea0 ...        rest-pose 椤剁偣
/// @param dSqr                杈?杈硅窛绂诲钩鏂?(鐢?_d_EE 璁＄畻)
/// @param barrier             barrier 鍙傛暟
/// @param kappa               鍒氬害绯绘暟
inline Real computeMollifiedBarrierEnergy(
    const glm::dvec3& ea0, const glm::dvec3& ea1,
    const glm::dvec3& eb0, const glm::dvec3& eb1,
    const glm::dvec3& rest_ea0, const glm::dvec3& rest_ea1,
    const glm::dvec3& rest_eb0, const glm::dvec3& rest_eb1,
    Real dSqr,
    const Barrier& barrier, Real kappa) {
  Real dHatSqr = barrier.dHatSqr();

  // I鈧?
  glm::dvec3 ea = ea1 - ea0;
  glm::dvec3 eb = eb1 - eb0;
  glm::dvec3 crossVec = glm::cross(ea, eb);
  Real I1 = glm::dot(crossVec, crossVec);
  if (I1 == 0.0) return 0.0;

  // I鈧?
  Real I2 = dSqr / dHatSqr;
  if (I2 >= 1.0 || I2 <= 0.0) return 0.0;

  // eps_x
  Real eps_x = computeEpsCross(rest_ea0, rest_ea1, rest_eb0, rest_eb1);

  // E = 魏 路 m(I鈧? 路 艥虏 路 (1-I鈧?虏 路 ln虏(I鈧?
  Real m = mollifierValue(I1, eps_x);
  Real L2 = std::log(I2);
  Real t2 = 1.0 - I2;
  Real sHat2 = dHatSqr * dHatSqr;
  return kappa * m * sHat2 * t2 * t2 * L2 * L2;
}

/// 鍒ゆ柇涓€瀵?EE 鏄惁闇€瑕佽蛋 mollifier 鍒嗘敮
///
/// @param ea0, ea1, eb0, eb1           褰撳墠浣嶇疆椤剁偣
/// @param rest_ea0, rest_ea1, ...      rest-pose 椤剁偣
/// @return true 琛ㄧず I鈧?< eps_x锛屽簲璧?mollifier 鍒嗘敮
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
// 鍐呴儴璇婃柇: 9脳9 鍐呭眰 Hessian (鐢ㄤ簬娴嬭瘯楠岃瘉)
// 鏋勯€?mollifier 鍒嗘敮鍦ㄥ瑙掑寲 vec(F) 绌洪棿涓殑 9脳9 SPD Hessian
// 杩欐槸涓€涓唴閮ㄥ嚱鏁帮紝姝ｅ父浠ｇ爜璺緞鐩存帴璋冪敤 computeMollifiedBarrier銆?
// ============================================================================

/// 鏋勯€?mollifier 鍒嗘敮鐨?9脳9 鍐呭眰 Hessian (SPD 鎶曞奖鍚?
/// @param I1     ||ea 脳 eb||虏 (鍏辩嚎搴?
/// @param I2     d虏/d虃虏 (褰掍竴鍖栬窛绂?
/// @param eps_x  rest-pose mollifier 闃堝€?
/// @param barrier  barrier 鍙傛暟
/// @param kappa    鍒氬害绯绘暟
inline Eigen::Matrix<Real, 9, 9> computeMollifierInnerHessian(
    Real I1, Real I2, Real eps_x,
    const Barrier& barrier, Real kappa) {
  Eigen::Matrix<Real, 9, 9> H = Eigen::Matrix<Real, 9, 9>::Zero();

  if (I1 <= 0 || I2 >= 1.0 || I2 <= 0) return H;

  // 鐗瑰緛鍊肩郴鏁?
  Real lambda10 = kappa * barrier.mollifierLambda10(I1, I2, eps_x);
  Real lambda11 = kappa * barrier.mollifierLambda11(I1, I2, eps_x);
  Real lambda12 = lambda11;  // 瀵圭О鎬?
  Real lambda20 = kappa * barrier.mollifierLambda20(I1, I2, eps_x);
  Real lambdag1g = kappa * barrier.mollifierLambdaG1G(I1, I2, eps_x);

  // === Twist 鏂瑰悜 (lambda11, lambda12) ===
  // 鍦?GIPC.cu 瀵硅鍖栧潗鏍囩郴涓?
  //   fnn = F 路 n1 路 n1岬€ = [[0,0,0],[0,c,0],[0,0,0]]
  //   q11 = normalize(vec(Tx 路 fnn)):
  //     Tx路fnn = [[0,0,0],[0,0,0],[0,-c/鈭?,0]]
  //     vec(col-major) = (0,0,0, 0,0,-c/鈭?, 0,0,0), 褰掍竴鍖?鈫?卤e鈧?
  //   q12 = normalize(vec(Tz 路 fnn)):
  //     Tz路fnn = [[0,c/鈭?,0],[0,0,0],[0,0,0]]
  //     vec(col-major) = (0,0,0, c/鈭?,0,0, 0,0,0), 褰掍竴鍖?鈫?卤e鈧?
  if (lambda11 > 0) {
    H(5, 5) += lambda11;
  }
  if (lambda12 > 0) {
    H(3, 3) += lambda12;
  }

  // === 璺濈-鍒囧悜鑰﹀悎 2脳2 瀛愰棶棰?===
  // {q10=e鈧? q20=e鈧坿 瀛愮┖闂?
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

} // namespace ksk::engine::contact::gipc
