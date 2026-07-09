// ============================================================================
// test-gpu-energy.cc — GpuEnergy::elastic vs closed-form CPU SNH energy.
//
// The GPU per-tet Stable Neo-Hookean energy density (reconstruct F = Ds*DmInv,
// Psi = 0.5*mu*(Ic-3) - mu*(J-1) + 0.5*lambda*(J-1)^2, times rest volume) summed
// by rpk::Reduce must match the same closed form evaluated on the CPU — and is
// the energy consistent with GpuElasticGradient/GpuElasticHessian's PK1/Hessian.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-energy.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cmath>
#include <vector>

using namespace sim::rhi;
using namespace sim::fem::gpu;

namespace {

struct EnergyFixture : public ::testing::Test {
    std::unique_ptr<Device>         device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<GPUEnergy>      energy;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        energy = std::make_unique<GPUEnergy>(*device, *compiler);
        if (!energy->valid()) GTEST_SKIP() << "pipeline failed to compile";
    }
};

// Closed-form CPU reference (same SNH density the GPU kernel evaluates).
double cpuElasticEnergy(const std::vector<glm::dvec3>& rest,
                        const std::vector<glm::dvec3>& cur,
                        const std::vector<std::array<int, 4>>& tets,
                        double mu, double lambda) {
    double sum = 0.0;
    for (const auto& t : tets) {
        glm::dvec3 r0 = rest[t[0]];
        glm::dmat3 Dm(rest[t[1]] - r0, rest[t[2]] - r0, rest[t[3]] - r0);
        glm::dmat3 DmInv = glm::inverse(Dm);
        double vol = glm::determinant(Dm) / 6.0;

        glm::dvec3 c0 = cur[t[0]];
        glm::dmat3 Ds(cur[t[1]] - c0, cur[t[2]] - c0, cur[t[3]] - c0);
        glm::dmat3 F = Ds * DmInv;

        double Ic = 0.0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) Ic += F[c][r] * F[c][r];
        double J = glm::determinant(F);
        double Jm1 = J - 1.0;
        double psi = 0.5 * mu * (Ic - 3.0) - mu * Jm1 + 0.5 * lambda * Jm1 * Jm1;
        sum += psi * vol;
    }
    return sum;
}

}  // namespace

TEST_F(EnergyFixture, ElasticMatchesClosedForm) {
    // Unit cube, 6-tet decomposition sharing the 0-6 diagonal.
    std::vector<glm::dvec3> rest = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    std::vector<std::array<int,4>> tets = {
        {0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}};

    // Current config: shear + stretch so F != I (non-trivial energy).
    std::vector<glm::dvec3> cur = rest;
    for (auto& p : cur) { p.x += 0.08 * p.y - 0.03 * p.z; p.y += 0.05 * p.z; p.z *= 1.04; }

    const double mu = 3.2, lambda = 7.5;

    double gpuE = energy->elastic(rest, cur, tets, mu, lambda);
    double cpuE = cpuElasticEnergy(rest, cur, tets, mu, lambda);

    spdlog::info("[test-gpu-energy] gpuE={:.10e} cpuE={:.10e} relErr={:.3e}",
                 gpuE, cpuE, std::abs(gpuE - cpuE) / std::max(std::abs(cpuE), 1e-30));
    ASSERT_GT(std::abs(cpuE), 0.0);
    EXPECT_LT(std::abs(gpuE - cpuE), 1e-9 * std::max(std::abs(cpuE), 1.0));

    // Rest configuration -> zero elastic energy.
    double gpuRest = energy->elastic(rest, rest, tets, mu, lambda);
    EXPECT_LT(std::abs(gpuRest), 1e-12);
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
