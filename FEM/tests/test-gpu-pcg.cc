// ============================================================================
// FEM/tests/test-gpu-pcg.cc
// Validates GpuBlockPCGSolver (BCOO, double) against CPU maths::BlockPCGSolver.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-block-pcg-solver.h>
#include <Maths/block-solvers/block-pcg.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <random>
#include <vector>

using namespace ksk;
using namespace ksk::fem::gpu;

namespace {

// Build a symmetric, diagonally-dominant (=> SPD) block matrix.
maths::BlockSparseMatrix<3> makeSpdBlockMatrix(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<int>     pick(0, n - 1);

    maths::BlockSparseMatrix<3> A(n, n);
    std::vector<glm::dmat3> diag(n, glm::dmat3(0.0));

    auto symBlock = [&](double scale) {
        glm::dmat3 M(0.0);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) M[c][r] = u(rng) * scale;
        return glm::dmat3(0.5) * (M + glm::transpose(M));  // symmetric
    };
    auto absRowSum = [](const glm::dmat3& M) {
        double s = 0.0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) s += std::abs(M[c][r]);
        return s;
    };

    // A few symmetric off-diagonal couplings per row.
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < 4; ++t) {
            int j = pick(rng);
            if (j <= i) continue;
            glm::dmat3 O = symBlock(0.3);
            A.addBlock(i, j, O);
            A.addBlock(j, i, O);  // O symmetric => transpose == O
            double s = absRowSum(O);
            diag[i] += glm::dmat3(s);
            diag[j] += glm::dmat3(s);
        }
    }
    // Diagonal blocks: positive-definite + dominance margin.
    for (int i = 0; i < n; ++i) {
        glm::dmat3 D = symBlock(0.2);
        D += glm::dmat3(8.0) + diag[i];  // ensure SPD via diagonal dominance
        A.addBlock(i, i, D);
    }
    return A;
}

struct GpuPcgFixture : public ::testing::Test {
    std::unique_ptr<ksk::rhi::Device> device;
    std::unique_ptr<ksk::rhi::ShaderCompiler> compiler;
    std::unique_ptr<GpuBlockPCGSolver> solver;

    void SetUp() override {
        device = ksk::rhi::Device::create({.backend = ksk::rhi::Backend::Vulkan,
                                           .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        solver = std::make_unique<GpuBlockPCGSolver>(*device, *compiler);
        if (!solver->valid()) GTEST_SKIP() << "PCG pipelines failed to compile";
    }
};

TEST_F(GpuPcgFixture, MatchesCpuSolver) {
    const int n = 256;
    auto A = makeSpdBlockMatrix(n, 12345u);

    maths::BlockVector<3> b(n);
    {
        std::mt19937 rng(999);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int i = 0; i < n; ++i) b[i] = glm::dvec3(u(rng), u(rng), u(rng));
    }

    // CPU reference
    maths::BlockVector<3> xCpu(n);
    xCpu.setZero();
    maths::BlockPCGSolver cpu(5000, 1e-12);
    auto cpuRes = cpu.solve(A, b, xCpu);
    ASSERT_TRUE(cpuRes.converged);

    // GPU
    maths::BlockVector<3> xGpu(n);
    xGpu.setZero();
    auto gpuRes = solver->solve(A, b, xGpu, 5000, 1e-12);
    EXPECT_TRUE(gpuRes.converged);

    // Compare solutions
    double maxErr = 0.0, refMax = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            maxErr = std::max(maxErr, std::abs(xGpu[i][d] - xCpu[i][d]));
            refMax = std::max(refMax, std::abs(xCpu[i][d]));
        }
    }
    double relErr = maxErr / std::max(refMax, 1e-30);
    spdlog::info("[test-gpu-pcg] gpuIters={} residual={:.3e} relErr={:.3e}",
                 gpuRes.iters, gpuRes.residual, relErr);
    EXPECT_LT(relErr, 1e-6);
}

} // namespace

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
