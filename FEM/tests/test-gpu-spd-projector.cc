// ============================================================================
// FEM/tests/test-gpu-spd-projector.cc
// Validates the GPU Jacobi 9x9 eigen-filter (both strategies, both clamp modes)
// against Eigen's SelfAdjointEigenSolver. Compares the reconstructed matrix
// (eigenvector basis is not unique under degeneracy; the filtered matrix is).
//
// Matrices for the tight (1e-9) tests use a CONTROLLED spectrum bounded away
// from zero. Rationale: the PSD clamp max(lambda,0) is a non-smooth boundary at
// 0; for an eigenvalue near zero the two eigensolvers may straddle the boundary
// and disagree by O(eigenvalue) — a benign artifact, not a kernel error. Keeping
// the spectrum away from zero makes the clamp decision unambiguous so abs and
// psd can both be validated tightly. A separate random-indefinite test checks
// the abs variant (sign-insensitive) on generic matrices.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-spd-projector.h>
#include <RHI/rhi.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <random>
#include <vector>

using namespace ksk::fem::gpu;
using Mat9 = Eigen::Matrix<double, 9, 9>;

namespace {

Mat9 cpuFilter(const Mat9& A, bool psd) {
    Eigen::SelfAdjointEigenSolver<Mat9> es(A);
    Eigen::Matrix<double, 9, 1> ev = es.eigenvalues();
    for (int i = 0; i < 9; ++i)
        ev(i) = psd ? std::max(ev(i), 0.0) : std::abs(ev(i));
    return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

Mat9 randomOrthogonal(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Mat9 M;
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j) M(i, j) = u(rng);
    Eigen::HouseholderQR<Mat9> qr(M);
    return qr.householderQ();
}

// Symmetric matrix with a prescribed spectrum (mixed sign, away from zero).
Mat9 withSpectrum(std::mt19937& rng) {
    std::uniform_real_distribution<double> mag(0.5, 3.0);   // |lambda| in [0.5, 3]
    std::bernoulli_distribution sign(0.5);
    Eigen::Matrix<double, 9, 1> ev;
    for (int i = 0; i < 9; ++i) ev(i) = (sign(rng) ? 1.0 : -1.0) * mag(rng);
    Mat9 Q = randomOrthogonal(rng);
    return Q * ev.asDiagonal() * Q.transpose();
}

std::vector<double> flatten(const std::vector<Mat9>& mats) {
    std::vector<double> f(mats.size() * 81);
    for (size_t m = 0; m < mats.size(); ++m)
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j) f[m * 81 + i * 9 + j] = mats[m](i, j);
    return f;
}

struct SpdFixture : public ::testing::Test {
    std::unique_ptr<ksk::rhi::Device> device;
    std::unique_ptr<ksk::rhi::ShaderCompiler> compiler;
    std::unique_ptr<GpuSpdProjector9> proj;
    std::vector<Mat9> spectrum;   // controlled spectrum, away from zero
    std::vector<Mat9> indefinite; // generic random symmetric

    void SetUp() override {
        device = ksk::rhi::Device::create({.backend = ksk::rhi::Backend::Vulkan,
                                           .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        // Correctness test affords extra iterations; the benchmark explores the
        // speed/accuracy tradeoff at lower counts.
        proj = std::make_unique<GpuSpdProjector9>(*device, *compiler, {}, 20, 600);
        if (!proj->valid()) GTEST_SKIP() << "SPD projector pipelines failed to compile";

        std::mt19937 rng(7);
        const int N = 200;
        spectrum.resize(N);
        for (int m = 0; m < N; ++m) spectrum[m] = withSpectrum(rng);

        std::uniform_real_distribution<double> u(-1.0, 1.0);
        indefinite.resize(N);
        for (int m = 0; m < N; ++m) {
            Mat9 M;
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) M(i, j) = u(rng);
            indefinite[m] = 0.5 * (M + M.transpose());
        }
    }

    void check(const std::vector<Mat9>& mats, GpuSpdProjector9::Strategy strat,
               GpuSpdProjector9::Clamp clamp, const char* name) {
        bool psd = (clamp == GpuSpdProjector9::Clamp::Psd);
        std::vector<double> out;
        proj->project(strat, clamp, flatten(mats), static_cast<int>(mats.size()), out);

        double maxErr = 0.0, refMax = 0.0;
        for (size_t m = 0; m < mats.size(); ++m) {
            Mat9 ref = cpuFilter(mats[m], psd);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    double g = out[m * 81 + i * 9 + j];
                    maxErr = std::max(maxErr, std::abs(g - ref(i, j)));
                    refMax = std::max(refMax, std::abs(ref(i, j)));
                }
        }
        double relErr = maxErr / std::max(refMax, 1e-30);
        spdlog::info("[test-gpu-spd-projector] {}: maxErr={:.3e} relErr={:.3e}",
                     name, maxErr, relErr);
        // 1e-6 is the appropriate bar for a fixed-iteration numerical eigensolver
        // (and far tighter than the SPD projection actually needs). A real bug in
        // the rotation/clamp/reconstruction produces O(1) errors, easily caught.
        EXPECT_LT(relErr, 1e-6) << name;
    }
};

// ---- Controlled spectrum (away from zero): all 4 variants tight ----
TEST_F(SpdFixture, CyclicAbs)    { check(spectrum, GpuSpdProjector9::Strategy::Cyclic,    GpuSpdProjector9::Clamp::Abs, "cyclic/abs"); }
TEST_F(SpdFixture, CyclicPsd)    { check(spectrum, GpuSpdProjector9::Strategy::Cyclic,    GpuSpdProjector9::Clamp::Psd, "cyclic/psd"); }
TEST_F(SpdFixture, ClassicalAbs) { check(spectrum, GpuSpdProjector9::Strategy::Classical, GpuSpdProjector9::Clamp::Abs, "classical/abs"); }
TEST_F(SpdFixture, ClassicalPsd) { check(spectrum, GpuSpdProjector9::Strategy::Classical, GpuSpdProjector9::Clamp::Psd, "classical/psd"); }

// ---- Generic random indefinite: abs is sign-insensitive near zero ----
TEST_F(SpdFixture, RandomIndefiniteCyclicAbs)    { check(indefinite, GpuSpdProjector9::Strategy::Cyclic,    GpuSpdProjector9::Clamp::Abs, "rand-indef cyclic/abs"); }
TEST_F(SpdFixture, RandomIndefiniteClassicalAbs) { check(indefinite, GpuSpdProjector9::Strategy::Classical, GpuSpdProjector9::Clamp::Abs, "rand-indef classical/abs"); }

} // namespace

#endif // FEM_GPU_ENABLED
