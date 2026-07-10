// ============================================================================
// FEM/tests/test-gpu-elastic-gradient.cc
// Validates GPU Stable Neo-Hookean elastic gradient against the CPU deform path
// (deform::StableNeoHookean + DeformationGradient::gradient, column-major vec).
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-elastic-gradient.h>
#include <Deform/strain-energy-density.h>
#include <Deform/deformation-gradient.h>
#include <Maths/tensor.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <random>
#include <vector>

using namespace ksk;
using namespace ksk::fem::gpu;

namespace {

Eigen::Vector3d toEigen(const glm::dvec3& v) { return {v.x, v.y, v.z}; }

TEST(GpuElasticGradient, MatchesCpuDeform) {
    auto device = ksk::rhi::Device::create({.backend = ksk::rhi::Backend::Vulkan,
                                            .enableValidation = true});
    if (!device) GTEST_SKIP() << "No Vulkan device";
    auto compiler = device->createShaderCompiler();
    if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

    GpuElasticGradient grad(*device, *compiler);
    if (!grad.valid()) GTEST_SKIP() << "elastic-gradient pipeline failed to compile";

    // ---- random rest + deformed config ----
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> pos(0.0, 1.0);
    std::uniform_real_distribution<double> def(-0.1, 0.1);

    const int N = 80;
    std::vector<glm::dvec3> rest(N), cur(N);
    for (int i = 0; i < N; ++i) {
        rest[i] = {pos(rng), pos(rng), pos(rng)};
        cur[i]  = rest[i] + glm::dvec3(def(rng), def(rng), def(rng));
    }

    // ---- random non-degenerate, positively-oriented tets ----
    std::uniform_int_distribution<int> pick(0, N - 1);
    std::vector<std::array<int, 4>> tets;
    while (tets.size() < 60) {
        std::array<int, 4> t{pick(rng), pick(rng), pick(rng), pick(rng)};
        if (t[0] == t[1] || t[0] == t[2] || t[0] == t[3] ||
            t[1] == t[2] || t[1] == t[3] || t[2] == t[3]) continue;
        maths::Matrix<double, 3, 3> Dm;
        for (int j = 0; j < 3; ++j) Dm.col(j) = toEigen(rest[t[j + 1]] - rest[t[0]]);
        double d = Dm.determinant();
        if (std::abs(d) < 1e-3) continue;
        if (d < 0) std::swap(t[0], t[1]);  // positive orientation
        tets.push_back(t);
    }

    // ---- material (use the adjusted mu/lambda from StableNeoHookean) ----
    deform::StableNeoHookean<double> snh(deform::ElasticityParameters<double>{1e5, 0.4});

    // ---- CPU reference (production deform path) ----
    std::vector<Eigen::Vector3d> ref(N, Eigen::Vector3d::Zero());
    for (const auto& t : tets) {
        maths::Matrix<double, 3, 3> Xm, xm;
        for (int j = 0; j < 3; ++j) {
            Xm.col(j) = toEigen(rest[t[j + 1]] - rest[t[0]]);
            xm.col(j) = toEigen(cur[t[j + 1]]  - cur[t[0]]);
        }
        deform::DeformationGradient<double, 3> dg(xm, Xm);
        auto P     = snh.computeEnergyGradient(dg);     // 3x3
        auto pFpx  = dg.gradient();                     // 9x12
        double vol = Xm.determinant() / 6.0;
        Eigen::Matrix<double, 12, 1> gx = pFpx.transpose() * maths::vectorize(P) * vol;
        for (int i = 0; i < 4; ++i) ref[t[i]] += gx.segment<3>(3 * i);
    }

    // ---- GPU ----
    std::vector<glm::dvec3> gpuGrad;
    grad.compute(rest, cur, tets, snh.mu, snh.lambda, gpuGrad);
    ASSERT_EQ(static_cast<int>(gpuGrad.size()), N);

    double maxErr = 0.0, refMax = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            maxErr = std::max(maxErr, std::abs(gpuGrad[i][d] - ref[i][d]));
            refMax = std::max(refMax, std::abs(ref[i][d]));
        }
    }
    double relErr = maxErr / std::max(refMax, 1e-30);
    spdlog::info("[test-gpu-elastic-gradient] maxErr={:.3e} relErr={:.3e}", maxErr, relErr);
    EXPECT_LT(relErr, 1e-9);
}

} // namespace

#endif // FEM_GPU_ENABLED
