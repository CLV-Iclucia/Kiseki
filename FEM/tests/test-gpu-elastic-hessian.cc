// ============================================================================
// FEM/tests/test-gpu-elastic-hessian.cc
// Validates GPU SNH elastic Hessian (assembly + Jacobi SPD filter + PFPx^T H
// PFPx) against the CPU deform path (filteredEnergyHessian). Compares the dense
// assembled 3N x 3N matrix (BCOO duplicates summed).
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-elastic-hessian.h>
#include <Deform/strain-energy-density.h>
#include <Deform/deformation-gradient.h>
#include <Maths/tensor.h>
#include <RHI/rhi.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <random>
#include <vector>

using namespace sim;
using namespace sim::fem::gpu;

namespace {

Eigen::Vector3d toEigen(const glm::dvec3& v) { return {v.x, v.y, v.z}; }

TEST(GpuElasticHessian, MatchesCpuDeform) {
    auto device = sim::rhi::Device::create({.backend = sim::rhi::Backend::Vulkan,
                                            .enableValidation = true});
    if (!device) GTEST_SKIP() << "No Vulkan device";
    auto compiler = device->createShaderCompiler();
    if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

    GpuElasticHessian hess(*device, *compiler, {}, /*cyclicSweeps=*/24);
    if (!hess.valid()) GTEST_SKIP() << "elastic-hessian pipeline failed to compile";

    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> pos(0.0, 1.0);
    std::uniform_real_distribution<double> def(-0.15, 0.15);

    const int N = 40;
    std::vector<glm::dvec3> rest(N), cur(N);
    for (int i = 0; i < N; ++i) {
        rest[i] = {pos(rng), pos(rng), pos(rng)};
        cur[i]  = rest[i] + glm::dvec3(def(rng), def(rng), def(rng));
    }

    std::uniform_int_distribution<int> pick(0, N - 1);
    std::vector<std::array<int, 4>> tets;
    while (tets.size() < 30) {
        std::array<int, 4> t{pick(rng), pick(rng), pick(rng), pick(rng)};
        if (t[0]==t[1]||t[0]==t[2]||t[0]==t[3]||t[1]==t[2]||t[1]==t[3]||t[2]==t[3]) continue;
        maths::Matrix<double, 3, 3> Dm;
        for (int j = 0; j < 3; ++j) Dm.col(j) = toEigen(rest[t[j + 1]] - rest[t[0]]);
        double d = Dm.determinant();
        if (std::abs(d) < 1e-3) continue;
        if (d < 0) std::swap(t[0], t[1]);
        tets.push_back(t);
    }

    deform::StableNeoHookean<double> snh(deform::ElasticityParameters<double>{1e5, 0.4});

    // ---- CPU reference: dense assembled Hessian ----
    Eigen::MatrixXd Hcpu = Eigen::MatrixXd::Zero(3 * N, 3 * N);
    for (const auto& t : tets) {
        maths::Matrix<double, 3, 3> Xm, xm;
        for (int j = 0; j < 3; ++j) {
            Xm.col(j) = toEigen(rest[t[j + 1]] - rest[t[0]]);
            xm.col(j) = toEigen(cur[t[j + 1]]  - cur[t[0]]);
        }
        deform::DeformationGradient<double, 3> dg(xm, Xm);
        auto Hf    = snh.filteredEnergyHessian(dg);   // 9x9
        auto pFpx  = dg.gradient();                   // 9x12
        double vol = Xm.determinant() / 6.0;
        Eigen::Matrix<double, 12, 12> Hx = pFpx.transpose() * Hf * pFpx * vol;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Hcpu.block<3, 3>(t[i] * 3, t[j] * 3) += Hx.block<3, 3>(i * 3, j * 3);
    }

    // ---- GPU: BCOO entries → dense ----
    std::vector<double> blocks; std::vector<uint32_t> row, col;
    hess.compute(rest, cur, tets, snh.mu, snh.lambda, blocks, row, col);

    Eigen::MatrixXd Hgpu = Eigen::MatrixXd::Zero(3 * N, 3 * N);
    for (size_t e = 0; e < row.size(); ++e)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Hgpu(row[e] * 3 + r, col[e] * 3 + c) += blocks[e * 9 + c * 3 + r];

    double maxErr = (Hgpu - Hcpu).cwiseAbs().maxCoeff();
    double refMax = Hcpu.cwiseAbs().maxCoeff();
    double relErr = maxErr / std::max(refMax, 1e-30);
    spdlog::info("[test-gpu-elastic-hessian] maxErr={:.3e} relErr={:.3e}", maxErr, relErr);
    EXPECT_LT(relErr, 1e-5);
}

} // namespace

#endif // FEM_GPU_ENABLED
