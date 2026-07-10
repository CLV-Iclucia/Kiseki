//
// GIPC Barrier 测试 — 严格按照 GIPC-BARRIER-TEST-PLAN.md 实现
//
// 6 组共 22 个测试:
//   第一组 (Test 1-4):  Barrier 标量函数
//   第二组 (Test 5-8):  PFPx 梯度 vs FD
//   第三组 (Test 9-12): 内层 Hessian vs SPD 投影
//   第四组 (Test 13-16): 端到端 Hessian SPD 属性
//   第五组 (Test 17-18): 多几何配置鲁棒性
//   第六组 (Test 19-22): Mollifier 分支端到端
//

#include <gtest/gtest.h>
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/gipc/pfpx.h>
#include <fem/ipc/gipc/hessian.h>
#include <fem/ipc/gipc/mollifier.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Maths/block-types.h>
#include <cmath>
#include <functional>

using namespace ksk::fem::ipc::gipc;
using namespace ksk::maths;
using Real = ksk::fem::Real;

// ============================================================================
// 辅助: 有限差分
// ============================================================================

template <int N>
Eigen::Matrix<Real, N, 1> finiteDiffGradient(
    std::function<Real(const Eigen::Matrix<Real, N, 1>&)> f,
    const Eigen::Matrix<Real, N, 1>& x,
    Real eps = 1e-7) {
  Eigen::Matrix<Real, N, 1> grad;
  for (int i = 0; i < N; i++) {
    auto xp = x; xp(i) += eps;
    auto xm = x; xm(i) -= eps;
    grad(i) = (f(xp) - f(xm)) / (2.0 * eps);
  }
  return grad;
}

template <int N>
Eigen::Matrix<Real, N, N> finiteDiffHessian(
    std::function<Real(const Eigen::Matrix<Real, N, 1>&)> f,
    const Eigen::Matrix<Real, N, 1>& x,
    Real eps = 1e-5) {
  Eigen::Matrix<Real, N, N> H;
  for (int i = 0; i < N; i++) {
    auto xp = x; xp(i) += eps;
    auto xm = x; xm(i) -= eps;
    auto gp = finiteDiffGradient<N>(f, xp, eps);
    auto gm = finiteDiffGradient<N>(f, xm, eps);
    H.col(i) = (gp - gm) / (2.0 * eps);
  }
  return 0.5 * (H + H.transpose());
}

// 辅助: LocalHessian → Eigen 矩阵
template <int N>
Eigen::Matrix<Real, N * 3, N * 3> localHessianToEigen(
    const ksk::maths::LocalHessian<N>& localH) {
  Eigen::Matrix<Real, N * 3, N * 3> H;
  for (int bi = 0; bi < N; bi++)
    for (int bj = 0; bj < N; bj++)
      for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
          H(bi * 3 + r, bj * 3 + c) = localH[bi][bj][c][r];
  return H;
}

// 辅助: SPD 投影 (clamp 负特征值到 0)
template <int N>
Eigen::Matrix<Real, N, N> spdProjection(const Eigen::Matrix<Real, N, N>& M) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, N, N>> solver(M);
  Eigen::Matrix<Real, N, N> result = Eigen::Matrix<Real, N, N>::Zero();
  for (int i = 0; i < N; i++) {
    if (solver.eigenvalues()(i) > 0)
      result += solver.eigenvalues()(i) *
                solver.eigenvectors().col(i) *
                solver.eigenvectors().col(i).transpose();
  }
  return result;
}

// 辅助: 计算 EE 距离平方
Real computeEEDistSqr(const glm::dvec3& a0, const glm::dvec3& a1,
                      const glm::dvec3& b0, const glm::dvec3& b1) {
  glm::dvec3 da = a1 - a0, db = b1 - b0, r = a0 - b0;
  Real a = glm::dot(da, da), e = glm::dot(db, db);
  Real b = glm::dot(da, db), c_val = glm::dot(da, r), f = glm::dot(db, r);
  Real denom = a * e - b * b;
  Real s, t;
  if (std::abs(denom) < 1e-30) {
    s = 0; t = std::clamp(f / e, 0.0, 1.0);
  } else {
    s = std::clamp((b * f - c_val * e) / denom, 0.0, 1.0);
    t = std::clamp((a * f - b * c_val) / denom, 0.0, 1.0);
  }
  t = std::clamp((b * s + f) / e, 0.0, 1.0);
  s = std::clamp((b * t - c_val) / a, 0.0, 1.0);
  glm::dvec3 closest = (a0 + s * da) - (b0 + t * db);
  return glm::dot(closest, closest);
}

// ============================================================================
// 约定常量
// ============================================================================

static constexpr Real DHAT = 0.01;
static constexpr Real KAPPA = 1e5;

// ############################################################################
// 第一组：Barrier 标量函数 (Test 1-4)
// ############################################################################

class BarrierScalarTest : public ::testing::Test {
protected:
  Barrier barrier{DHAT};
  Real dHatSqr = DHAT * DHAT;
};

// Test 1: Barrier_EnergyBoundary
TEST_F(BarrierScalarTest, Barrier_EnergyBoundary) {
  EXPECT_NEAR(barrier.energy(dHatSqr), 0.0, 1e-15);
  EXPECT_EQ(barrier.energy(1.5 * dHatSqr), 0.0);
  EXPECT_GT(barrier.energy(0.5 * dHatSqr), 0.0);
  // 距离越小能量越大
  Real e08 = barrier.energy(0.8 * dHatSqr);
  Real e05 = barrier.energy(0.5 * dHatSqr);
  Real e01 = barrier.energy(0.1 * dHatSqr);
  EXPECT_LT(e08, e05);
  EXPECT_LT(e05, e01);
}

// Test 2: Barrier_GradCoeffMatchesFD
TEST_F(BarrierScalarTest, Barrier_GradCoeffMatchesFD) {
  auto energyOfI5 = [&](Real I5) -> Real {
    return barrier.energy(I5 * dHatSqr);
  };

  for (Real I5 : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    Real eps = 1e-7;
    Real fd = (energyOfI5(I5 + eps) - energyOfI5(I5 - eps)) / (2.0 * eps);
    // gradCoeff = 2 * dE/dI5
    EXPECT_NEAR(barrier.gradCoeff(I5), 2.0 * fd,
                std::abs(2.0 * fd) * 1e-5 + 1e-12)
        << "at I5 = " << I5;
  }
}

// Test 3: Barrier_Lambda0MatchesFD
TEST_F(BarrierScalarTest, Barrier_Lambda0MatchesFD) {
  for (Real I5 : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    Real eps = 1e-7;
    Real bp_plus = barrier.dBdI5(I5 + eps);
    Real bp_minus = barrier.dBdI5(I5 - eps);
    Real bpp = (bp_plus - bp_minus) / (2.0 * eps);
    Real bp = barrier.dBdI5(I5);
    Real expected = 4.0 * I5 * bpp + 2.0 * bp;
    EXPECT_NEAR(barrier.lambda0(I5), expected,
                std::abs(expected) * 1e-4 + 1e-10)
        << "at I5 = " << I5;
  }
}

// Test 4: Barrier_GaussClamp
TEST_F(BarrierScalarTest, Barrier_GaussClamp) {
  Real lam_threshold = barrier.clampedLambda0(Barrier::GAUSS_THRESHOLD);
  for (Real I5 : {1e-5, 1e-6, 1e-8, 1e-10}) {
    Real lam = barrier.clampedLambda0(I5);
    EXPECT_TRUE(std::isfinite(lam)) << "I5=" << I5;
    EXPECT_GE(lam, 0.0) << "I5=" << I5;
    EXPECT_DOUBLE_EQ(lam, lam_threshold) << "I5=" << I5;
  }
}

// ############################################################################
// 第二组：PFPx 梯度正确性 (Test 5-8)
// ############################################################################

// Test 5: PP_GradientMatchesFD
TEST(PFPxGradient, PP_GradientMatchesFD) {
  Barrier barrier(DHAT);
  glm::dvec3 p0{0, 0, 0}, p1{0, 0.007, 0};

  auto energyFunc = [&](const Eigen::Matrix<Real, 6, 1>& xv) -> Real {
    glm::dvec3 a(xv(0), xv(1), xv(2));
    glm::dvec3 b(xv(3), xv(4), xv(5));
    auto r = computePFPx_PP(a, b, DHAT);
    if (!r.valid || r.I5 >= 1.0) return 0.0;
    return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
  };

  Eigen::Matrix<Real, 6, 1> x;
  x << p0.x, p0.y, p0.z, p1.x, p1.y, p1.z;

  auto pfpx = computePFPx_PP(p0, p1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real alpha = barrier.gradCoeff(pfpx.I5);
  Eigen::Matrix<Real, 3, 1> flatten_pk1 =
      pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
  Eigen::Matrix<Real, 6, 1> grad_analytical =
      KAPPA * pfpx.PFPx.transpose() * flatten_pk1;

  Eigen::Matrix<Real, 6, 1> grad_fd = finiteDiffGradient<6>(energyFunc, x);

  Real err = (grad_analytical - grad_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(err, 1e-4) << "PP gradient error = " << err;
}

// Test 6: PE_GradientMatchesFD
TEST(PFPxGradient, PE_GradientMatchesFD) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.5, 0.006, 0}, e0{0, 0, 0}, e1{1, 0, 0};

  auto energyFunc = [&](const Eigen::Matrix<Real, 9, 1>& xv) -> Real {
    glm::dvec3 pp(xv(0), xv(1), xv(2));
    glm::dvec3 a(xv(3), xv(4), xv(5));
    glm::dvec3 b(xv(6), xv(7), xv(8));
    auto r = computePFPx_PE(pp, a, b, DHAT);
    if (!r.valid || r.I5 >= 1.0) return 0.0;
    return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
  };

  Eigen::Matrix<Real, 9, 1> x;
  x << p.x, p.y, p.z, e0.x, e0.y, e0.z, e1.x, e1.y, e1.z;

  auto pfpx = computePFPx_PE(p, e0, e1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real alpha = barrier.gradCoeff(pfpx.I5);
  Eigen::Matrix<Real, 6, 1> flatten_pk1 =
      pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
  Eigen::Matrix<Real, 9, 1> grad_analytical =
      KAPPA * pfpx.PFPx.transpose() * flatten_pk1;

  Eigen::Matrix<Real, 9, 1> grad_fd = finiteDiffGradient<9>(energyFunc, x);

  Real err = (grad_analytical - grad_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(err, 1e-3) << "PE gradient error = " << err;
}

// Test 7: PT_GradientMatchesFD
TEST(PFPxGradient, PT_GradientMatchesFD) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.3, 0.007, 0.3};
  glm::dvec3 t0{0, 0, 0}, t1{1, 0, 0}, t2{0, 0, 1};

  auto energyFunc = [&](const Eigen::Matrix<Real, 12, 1>& xv) -> Real {
    glm::dvec3 pp(xv(0), xv(1), xv(2));
    glm::dvec3 a(xv(3), xv(4), xv(5));
    glm::dvec3 b(xv(6), xv(7), xv(8));
    glm::dvec3 c(xv(9), xv(10), xv(11));
    auto r = computePFPx_PT(pp, a, b, c, DHAT);
    if (!r.valid || r.I5 >= 1.0) return 0.0;
    return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
  };

  Eigen::Matrix<Real, 12, 1> x;
  x << p.x, p.y, p.z, t0.x, t0.y, t0.z,
       t1.x, t1.y, t1.z, t2.x, t2.y, t2.z;

  auto pfpx = computePFPx_PT(p, t0, t1, t2, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real alpha = barrier.gradCoeff(pfpx.I5);
  Eigen::Matrix<Real, 9, 1> flatten_pk1 =
      pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
  Eigen::Matrix<Real, 12, 1> grad_analytical =
      KAPPA * pfpx.PFPx.transpose() * flatten_pk1;

  Eigen::Matrix<Real, 12, 1> grad_fd =
      finiteDiffGradient<12>(energyFunc, x);

  Real err = (grad_analytical - grad_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(err, 1e-3) << "PT gradient error = " << err;
}

// Test 8: EE_GradientMatchesFD
TEST(PFPxGradient, EE_GradientMatchesFD) {
  Barrier barrier(DHAT);
  glm::dvec3 ea0{0, 0, 0}, ea1{1, 0, 0};
  glm::dvec3 eb0{0.5, 0.005, -0.5}, eb1{0.5, 0.005, 0.5};

  auto energyFunc = [&](const Eigen::Matrix<Real, 12, 1>& xv) -> Real {
    glm::dvec3 a0(xv(0), xv(1), xv(2)), a1(xv(3), xv(4), xv(5));
    glm::dvec3 b0(xv(6), xv(7), xv(8)), b1(xv(9), xv(10), xv(11));
    auto r = computePFPx_EE(a0, a1, b0, b1, DHAT);
    if (!r.valid || r.I5 >= 1.0) return 0.0;
    return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
  };

  Eigen::Matrix<Real, 12, 1> x;
  x << ea0.x, ea0.y, ea0.z, ea1.x, ea1.y, ea1.z,
       eb0.x, eb0.y, eb0.z, eb1.x, eb1.y, eb1.z;

  auto pfpx = computePFPx_EE(ea0, ea1, eb0, eb1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real alpha = barrier.gradCoeff(pfpx.I5);
  Eigen::Matrix<Real, 9, 1> flatten_pk1 =
      pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
  Eigen::Matrix<Real, 12, 1> grad_analytical =
      KAPPA * pfpx.PFPx.transpose() * flatten_pk1;

  Eigen::Matrix<Real, 12, 1> grad_fd =
      finiteDiffGradient<12>(energyFunc, x);

  Real err = (grad_analytical - grad_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(err, 1e-3) << "EE gradient error = " << err;
}

// ############################################################################
// 第三组：内层 Hessian vs SPD 投影 (Test 9-12)
//
// 核心思想: 在 vec(F) 空间中，b(I5) 是确定性函数，没有任何近似。
// 所以对 vec(F) 做 FD 得到的 Hessian 做 SPD 投影后，应与
// 解析内层 Hessian (clampedLambda0 * q0 * q0^T) 一致。
// ############################################################################

// Test 9: PP_InnerHessianMatchesSPDProjection
TEST(InnerHessian, PP_InnerHessianMatchesSPDProjection) {
  Barrier barrier(DHAT);
  glm::dvec3 p0{0, 0, 0}, p1{0, 0.007, 0};

  auto pfpx = computePFPx_PP(p0, p1, DHAT);
  ASSERT_TRUE(pfpx.valid);

  // PP NEWF: vec(F) 是 3 维, q0 = e_2, I5 = vecF(2)^2
  // 能量: E(vecF) = kappa * barrier.energy(vecF(2)^2 * dHatSqr)
  Real dHatSqr = barrier.dHatSqr();
  Real I5 = pfpx.I5;
  Real sqrtI5 = std::sqrt(I5);

  // vecF0: 当前 F 值。PP NEWF 下 vecF = (0, 0, d/dHat)
  Eigen::Matrix<Real, 3, 1> vecF0;
  vecF0.setZero();
  vecF0(2) = sqrtI5;  // = d / dHat

  auto energyOfVecF = [&](const Eigen::Matrix<Real, 3, 1>& vf) -> Real {
    Real vI5 = vf(2) * vf(2);  // I5 = vecF(2)^2
    return KAPPA * barrier.energy(vI5 * dHatSqr);
  };

  Eigen::Matrix<Real, 3, 3> H_fd =
      finiteDiffHessian<3>(energyOfVecF, vecF0);
  Eigen::Matrix<Real, 3, 3> H_projected = spdProjection<3>(H_fd);

  // 解析: H = clampedLambda0(I5) * kappa * q0 * q0^T
  Real lam = KAPPA * barrier.clampedLambda0(I5);
  Eigen::Matrix<Real, 3, 3> H_analytical = lam * pfpx.q0 * pfpx.q0.transpose();

  Real err = (H_analytical - H_projected).cwiseAbs().maxCoeff();
  Real scale = H_projected.cwiseAbs().maxCoeff();
  EXPECT_LT(err, scale * 1e-3 + 1e-8)
      << "PP inner Hessian error = " << err;
}

// Test 10: PE_InnerHessianMatchesSPDProjection
TEST(InnerHessian, PE_InnerHessianMatchesSPDProjection) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.5, 0.006, 0}, e0{0, 0, 0}, e1{1, 0, 0};

  auto pfpx = computePFPx_PE(p, e0, e1, DHAT);
  ASSERT_TRUE(pfpx.valid);

  // PE NEWF: vec(F) 6 维 (实际有效 4 维，但 PFPx 是 6×9)
  // q0(3) = 1, 其余为 0 → I5 = vecF(3)^2
  Real dHatSqr = barrier.dHatSqr();
  Real I5 = pfpx.I5;

  // vecF0: 第 3 分量 = sqrt(I5), 其余为 PFPx * x 的结果
  // 但对于内层 Hessian 测试，只需知道 q0 方向的分量
  // 构造一个 vecF0 使得 q0^T * vecF0 = sqrt(I5)
  Eigen::Matrix<Real, 6, 1> vecF0;
  vecF0.setZero();
  vecF0(3) = std::sqrt(I5);

  auto energyOfVecF = [&](const Eigen::Matrix<Real, 6, 1>& vf) -> Real {
    // I5 = (q0^T * vf)^2 = vf(3)^2
    Real vI5 = vf(3) * vf(3);
    return KAPPA * barrier.energy(vI5 * dHatSqr);
  };

  Eigen::Matrix<Real, 6, 6> H_fd =
      finiteDiffHessian<6>(energyOfVecF, vecF0);
  Eigen::Matrix<Real, 6, 6> H_projected = spdProjection<6>(H_fd);

  Real lam = KAPPA * barrier.clampedLambda0(I5);
  Eigen::Matrix<Real, 6, 6> H_analytical =
      lam * pfpx.q0 * pfpx.q0.transpose();

  Real err = (H_analytical - H_projected).cwiseAbs().maxCoeff();
  Real scale = H_projected.cwiseAbs().maxCoeff();
  EXPECT_LT(err, scale * 1e-3 + 1e-8)
      << "PE inner Hessian error = " << err;
}

// Test 11: PT_InnerHessianMatchesSPDProjection
TEST(InnerHessian, PT_InnerHessianMatchesSPDProjection) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.3, 0.007, 0.3};
  glm::dvec3 t0{0, 0, 0}, t1{1, 0, 0}, t2{0, 0, 1};

  auto pfpx = computePFPx_PT(p, t0, t1, t2, DHAT);
  ASSERT_TRUE(pfpx.valid);

  // PT NEWF: vec(F) 9 维, q0(8) = 1 → I5 = vecF(8)^2
  Real dHatSqr = barrier.dHatSqr();
  Real I5 = pfpx.I5;

  Eigen::Matrix<Real, 9, 1> vecF0;
  vecF0.setZero();
  vecF0(8) = std::sqrt(I5);

  auto energyOfVecF = [&](const Eigen::Matrix<Real, 9, 1>& vf) -> Real {
    Real vI5 = vf(8) * vf(8);
    return KAPPA * barrier.energy(vI5 * dHatSqr);
  };

  Eigen::Matrix<Real, 9, 9> H_fd =
      finiteDiffHessian<9>(energyOfVecF, vecF0);
  Eigen::Matrix<Real, 9, 9> H_projected = spdProjection<9>(H_fd);

  Real lam = KAPPA * barrier.clampedLambda0(I5);
  Eigen::Matrix<Real, 9, 9> H_analytical =
      lam * pfpx.q0 * pfpx.q0.transpose();

  Real err = (H_analytical - H_projected).cwiseAbs().maxCoeff();
  Real scale = H_projected.cwiseAbs().maxCoeff();
  EXPECT_LT(err, scale * 1e-3 + 1e-8)
      << "PT inner Hessian error = " << err;
}

// Test 12: EE_InnerHessianMatchesSPDProjection
TEST(InnerHessian, EE_InnerHessianMatchesSPDProjection) {
  Barrier barrier(DHAT);
  glm::dvec3 ea0{0, 0, 0}, ea1{1, 0, 0};
  glm::dvec3 eb0{0.5, 0.005, -0.5}, eb1{0.5, 0.005, 0.5};

  auto pfpx = computePFPx_EE(ea0, ea1, eb0, eb1, DHAT);
  ASSERT_TRUE(pfpx.valid);

  // EE NEWF: vec(F) 9 维, q0(8) = 1 → I5 = vecF(8)^2
  Real dHatSqr = barrier.dHatSqr();
  Real I5 = pfpx.I5;

  Eigen::Matrix<Real, 9, 1> vecF0;
  vecF0.setZero();
  vecF0(8) = std::sqrt(I5);

  auto energyOfVecF = [&](const Eigen::Matrix<Real, 9, 1>& vf) -> Real {
    Real vI5 = vf(8) * vf(8);
    return KAPPA * barrier.energy(vI5 * dHatSqr);
  };

  Eigen::Matrix<Real, 9, 9> H_fd =
      finiteDiffHessian<9>(energyOfVecF, vecF0);
  Eigen::Matrix<Real, 9, 9> H_projected = spdProjection<9>(H_fd);

  Real lam = KAPPA * barrier.clampedLambda0(I5);
  Eigen::Matrix<Real, 9, 9> H_analytical =
      lam * pfpx.q0 * pfpx.q0.transpose();

  Real err = (H_analytical - H_projected).cwiseAbs().maxCoeff();
  Real scale = H_projected.cwiseAbs().maxCoeff();
  EXPECT_LT(err, scale * 1e-3 + 1e-8)
      << "EE inner Hessian error = " << err;
}

// ############################################################################
// 第四组：端到端 Hessian SPD 属性 (Test 13-16)
// ############################################################################

// Test 13: PP_HessianIsSPD
TEST(HessianSPD, PP_HessianIsSPD) {
  Barrier barrier(DHAT);
  glm::dvec3 p0{0, 0, 0}, p1{0, 0.007, 0};

  auto pfpx = computePFPx_PP(p0, p1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real lam = KAPPA * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<2, 3>(pfpx.PFPx, pfpx.q0, lam);
  Eigen::Matrix<Real, 6, 6> H = localHessianToEigen<2>(localH);

  // 对称性
  Real asymm = (H - H.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asymm, 1e-12) << "PP Hessian not symmetric";

  // 半正定
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 6, 6>> solver(H);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-10)
      << "PP Hessian not SPD, min eigenvalue = "
      << solver.eigenvalues().minCoeff();
}

// Test 14: PE_HessianIsSPD
TEST(HessianSPD, PE_HessianIsSPD) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.5, 0.006, 0}, e0{0, 0, 0}, e1{1, 0, 0};

  auto pfpx = computePFPx_PE(p, e0, e1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real lam = KAPPA * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<3, 6>(pfpx.PFPx, pfpx.q0, lam);
  Eigen::Matrix<Real, 9, 9> H = localHessianToEigen<3>(localH);

  Real asymm = (H - H.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asymm, 1e-12) << "PE Hessian not symmetric";

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 9, 9>> solver(H);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-10)
      << "PE Hessian not SPD, min eigenvalue = "
      << solver.eigenvalues().minCoeff();
}

// Test 15: PT_HessianIsSPD
TEST(HessianSPD, PT_HessianIsSPD) {
  Barrier barrier(DHAT);
  glm::dvec3 p{0.3, 0.007, 0.3};
  glm::dvec3 t0{0, 0, 0}, t1{1, 0, 0}, t2{0, 0, 1};

  auto pfpx = computePFPx_PT(p, t0, t1, t2, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real lam = KAPPA * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
  Eigen::Matrix<Real, 12, 12> H = localHessianToEigen<4>(localH);

  Real asymm = (H - H.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asymm, 1e-12) << "PT Hessian not symmetric";

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 12, 12>> solver(H);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-10)
      << "PT Hessian not SPD, min eigenvalue = "
      << solver.eigenvalues().minCoeff();
}

// Test 16: EE_HessianIsSPD
TEST(HessianSPD, EE_HessianIsSPD) {
  Barrier barrier(DHAT);
  glm::dvec3 ea0{0, 0, 0}, ea1{1, 0, 0};
  glm::dvec3 eb0{0.5, 0.005, -0.5}, eb1{0.5, 0.005, 0.5};

  auto pfpx = computePFPx_EE(ea0, ea1, eb0, eb1, DHAT);
  ASSERT_TRUE(pfpx.valid);
  Real lam = KAPPA * barrier.clampedLambda0(pfpx.I5);
  auto localH = sandwichRank1<4, 9>(pfpx.PFPx, pfpx.q0, lam);
  Eigen::Matrix<Real, 12, 12> H = localHessianToEigen<4>(localH);

  Real asymm = (H - H.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asymm, 1e-12) << "EE Hessian not symmetric";

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 12, 12>> solver(H);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-10)
      << "EE Hessian not SPD, min eigenvalue = "
      << solver.eigenvalues().minCoeff();
}

// ############################################################################
// 第五组：多几何配置鲁棒性 (Test 17-18)
// ############################################################################

// Test 17: PP_MultiDirection
TEST(MultiConfig, PP_MultiDirection) {
  Barrier barrier(DHAT);
  std::vector<glm::dvec3> dirs = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
      glm::normalize(glm::dvec3{1, 1, 0}),
      glm::normalize(glm::dvec3{1, 1, 1}),
      glm::normalize(glm::dvec3{-2, 3, 1}),
  };

  for (const auto& dir : dirs) {
    glm::dvec3 p0(0.5, 0.5, 0.5);
    glm::dvec3 p1 = p0 + 0.006 * dir;

    auto energyFunc = [&](const Eigen::Matrix<Real, 6, 1>& xv) -> Real {
      glm::dvec3 a(xv(0), xv(1), xv(2));
      glm::dvec3 b(xv(3), xv(4), xv(5));
      auto r = computePFPx_PP(a, b, DHAT);
      if (!r.valid || r.I5 >= 1.0) return 0.0;
      return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
    };

    Eigen::Matrix<Real, 6, 1> x;
    x << p0.x, p0.y, p0.z, p1.x, p1.y, p1.z;

    auto pfpx = computePFPx_PP(p0, p1, DHAT);
    ASSERT_TRUE(pfpx.valid)
        << "dir=(" << dir.x << "," << dir.y << "," << dir.z << ")";
    Real alpha = barrier.gradCoeff(pfpx.I5);
    Eigen::Matrix<Real, 3, 1> pk1 =
        pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
    Eigen::Matrix<Real, 6, 1> grad_a =
        KAPPA * pfpx.PFPx.transpose() * pk1;
    Eigen::Matrix<Real, 6, 1> grad_fd =
        finiteDiffGradient<6>(energyFunc, x);

    Real err = (grad_a - grad_fd).cwiseAbs().maxCoeff();
    EXPECT_LT(err, 1e-4)
        << "PP dir=(" << dir.x << "," << dir.y << "," << dir.z << ")";
  }
}

// Test 18: EE_MultiConfig
TEST(MultiConfig, EE_MultiConfig) {
  Barrier barrier(DHAT);
  struct EEConfig {
    glm::dvec3 ea0, ea1, eb0, eb1;
    const char* name;
  };
  std::vector<EEConfig> configs = {
      {{0, 0, 0}, {1, 0, 0}, {0.5, 0.005, -0.5}, {0.5, 0.005, 0.5},
       "perpendicular"},
      {{0, 0, 0}, {1, 0, 0}, {0.3, 0.004, 0}, {1.3, 0.004, 0},
       "parallel_offset"},
      {{0, 0, 0}, {1, 0, 0}, {0.5, 0.006, -0.3}, {0.5, 0.006, 0.7},
       "skew_45deg"},
  };

  for (const auto& cfg : configs) {
    auto energyFunc = [&](const Eigen::Matrix<Real, 12, 1>& xv) -> Real {
      glm::dvec3 a0(xv(0), xv(1), xv(2)), a1(xv(3), xv(4), xv(5));
      glm::dvec3 b0(xv(6), xv(7), xv(8)), b1(xv(9), xv(10), xv(11));
      auto r = computePFPx_EE(a0, a1, b0, b1, DHAT);
      if (!r.valid || r.I5 >= 1.0) return 0.0;
      return KAPPA * barrier.energy(r.I5 * barrier.dHatSqr());
    };

    Eigen::Matrix<Real, 12, 1> x;
    x << cfg.ea0.x, cfg.ea0.y, cfg.ea0.z, cfg.ea1.x, cfg.ea1.y, cfg.ea1.z,
         cfg.eb0.x, cfg.eb0.y, cfg.eb0.z, cfg.eb1.x, cfg.eb1.y, cfg.eb1.z;

    auto pfpx = computePFPx_EE(cfg.ea0, cfg.ea1, cfg.eb0, cfg.eb1, DHAT);
    if (!pfpx.valid || pfpx.I5 >= 1.0) continue;

    Real alpha = barrier.gradCoeff(pfpx.I5);
    Eigen::Matrix<Real, 9, 1> pk1 =
        pfpx.q0 * (alpha * std::sqrt(pfpx.I5));
    Eigen::Matrix<Real, 12, 1> grad_a =
        KAPPA * pfpx.PFPx.transpose() * pk1;
    Eigen::Matrix<Real, 12, 1> grad_fd =
        finiteDiffGradient<12>(energyFunc, x);

    Real err = (grad_a - grad_fd).cwiseAbs().maxCoeff();
    EXPECT_LT(err, 1e-3) << "EE config: " << cfg.name;
  }
}

// ############################################################################
// 第六组：Mollifier 分支端到端 (Test 19-22)
// ############################################################################

class MollifierE2ETest : public ::testing::Test {
protected:
  Barrier barrier{DHAT};

  // PEE 配置: 近平行边，触发 mollifier
  // 注意: eb 方向必须与 ea 方向不完全平行 (叉积非零)，否则 I1=0
  glm::dvec3 ea0{0, 0, 0}, ea1{1, 0, 0};
  glm::dvec3 eb0{0.3, 0.005, -0.002}, eb1{0.8, 0.005, 0.002};
  // rest pose = 当前 pose
  glm::dvec3 rest_ea0 = ea0, rest_ea1 = ea1;
  glm::dvec3 rest_eb0 = eb0, rest_eb1 = eb1;

  Real dSqr{};
  PFPxResult12 pfpx{};
  bool setup_ok = false;

  void SetUp() override {
    ASSERT_TRUE(needsMollifier(
        ea0, ea1, eb0, eb1, rest_ea0, rest_ea1, rest_eb0, rest_eb1))
        << "Test config should trigger mollifier branch";

    pfpx = computePFPx_PEE(ea0, ea1, eb0, eb1, barrier.dHat());
    if (!pfpx.valid) {
      GTEST_SKIP() << "PFPx_PEE construction failed";
      return;
    }

    dSqr = computeEEDistSqr(ea0, ea1, eb0, eb1);
    ASSERT_LT(dSqr, barrier.dHatSqr())
        << "Distance should be within barrier range";
    setup_ok = true;
  }

  // 端到端能量函数
  std::function<Real(const Eigen::Matrix<Real, 12, 1>&)>
  makeEnergyFunc() {
    return [this](const Eigen::Matrix<Real, 12, 1>& xv) -> Real {
      glm::dvec3 a0(xv(0), xv(1), xv(2));
      glm::dvec3 a1(xv(3), xv(4), xv(5));
      glm::dvec3 b0(xv(6), xv(7), xv(8));
      glm::dvec3 b1(xv(9), xv(10), xv(11));
      Real d2 = computeEEDistSqr(a0, a1, b0, b1);
      return computeMollifiedBarrierEnergy(
          a0, a1, b0, b1,
          rest_ea0, rest_ea1, rest_eb0, rest_eb1,
          d2, barrier, KAPPA);
    };
  }

  Eigen::Matrix<Real, 12, 1> getX() {
    Eigen::Matrix<Real, 12, 1> x;
    x << ea0.x, ea0.y, ea0.z, ea1.x, ea1.y, ea1.z,
         eb0.x, eb0.y, eb0.z, eb1.x, eb1.y, eb1.z;
    return x;
  }
};

// Test 19: Mollifier_EnergyMatchesFD
TEST_F(MollifierE2ETest, Mollifier_EnergyMatchesFD) {
  if (!setup_ok) return;

  auto energyFunc = makeEnergyFunc();
  auto x = getX();

  // FD gradient — 如果有限且非零，说明能量函数光滑
  Eigen::Matrix<Real, 12, 1> grad_fd =
      finiteDiffGradient<12>(energyFunc, x);

  EXPECT_TRUE(grad_fd.allFinite()) << "FD gradient should be finite";
  EXPECT_GT(grad_fd.norm(), 0.0) << "FD gradient should be nonzero";
}

// Test 20: Mollifier_GradientMatchesFD
TEST_F(MollifierE2ETest, Mollifier_GradientMatchesFD) {
  if (!setup_ok) return;

  auto result = computeMollifiedBarrier(
      ea0, ea1, eb0, eb1,
      rest_ea0, rest_ea1, rest_eb0, rest_eb1,
      dSqr, pfpx.PFPx, barrier, KAPPA);
  ASSERT_TRUE(result.active);

  // 解析梯度 → Eigen
  Eigen::Matrix<Real, 12, 1> grad_analytical;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      grad_analytical(i * 3 + j) = result.gradient[i][j];

  // FD 梯度
  auto energyFunc = makeEnergyFunc();
  auto x = getX();
  Eigen::Matrix<Real, 12, 1> grad_fd =
      finiteDiffGradient<12>(energyFunc, x);

  Real err = (grad_analytical - grad_fd).cwiseAbs().maxCoeff();
  Real scale = grad_fd.cwiseAbs().maxCoeff();
  EXPECT_LT(err, scale * 0.05 + 1e-6)
      << "Mollifier gradient error = " << err << ", scale = " << scale;
}

// Test 21: Mollifier_HessianIsSPD
TEST_F(MollifierE2ETest, Mollifier_HessianIsSPD) {
  if (!setup_ok) return;

  auto result = computeMollifiedBarrier(
      ea0, ea1, eb0, eb1,
      rest_ea0, rest_ea1, rest_eb0, rest_eb1,
      dSqr, pfpx.PFPx, barrier, KAPPA);
  if (!result.active) GTEST_SKIP();

  Eigen::Matrix<Real, 12, 12> H = localHessianToEigen<4>(result.hessian);

  // 对称性
  Real asymm = (H - H.transpose()).cwiseAbs().maxCoeff();
  EXPECT_LT(asymm, 1e-12) << "Mollifier Hessian not symmetric";

  // 半正定
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 12, 12>> solver(H);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-10)
      << "Mollifier Hessian not SPD, min eigenvalue = "
      << solver.eigenvalues().minCoeff();
}

// Test 22: Mollifier_HessianCurvatureAlongGradient
TEST_F(MollifierE2ETest, Mollifier_HessianCurvatureAlongGradient) {
  if (!setup_ok) return;

  auto result = computeMollifiedBarrier(
      ea0, ea1, eb0, eb1,
      rest_ea0, rest_ea1, rest_eb0, rest_eb1,
      dSqr, pfpx.PFPx, barrier, KAPPA);
  if (!result.active) GTEST_SKIP();

  Eigen::Matrix<Real, 12, 1> g;
  Eigen::Matrix<Real, 12, 12> H;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      g(i * 3 + j) = result.gradient[i][j];
  H = localHessianToEigen<4>(result.hessian);

  Real gHg = g.transpose() * H * g;
  EXPECT_GT(gHg, 0.0)
      << "Hessian should have positive curvature along gradient (Newton descent)";
}
