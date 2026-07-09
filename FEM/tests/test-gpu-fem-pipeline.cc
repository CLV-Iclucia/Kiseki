// ============================================================================
// test-gpu-fem-pipeline.cc — end-to-end GPU chain:
//   GpuElasticHessian (device BCOO) -> GpuBcooSorter -> GpuBlockPCGSolver::solveDevice
//
// The matrix never round-trips through the host: assembly leaves the BCOO in
// device buffers, a mass diagonal is appended (device copy + small upload), the
// sorter orders it by row, and the PCG solver consumes the device buffers
// directly. Correctness is checked by solving the *same* (GPU-assembled) system
// on the CPU with maths::BlockPCGSolver. The single download in the test exists
// only to build that reference and is not part of the pipeline under test.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-elastic-hessian.h>
#include <fem/gpu/gpu-bcoo-sorter.h>
#include <fem/gpu/gpu-block-pcg-solver.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-vector.h>
#include <Maths/block-solvers/block-pcg.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace sim;
using namespace sim::rhi;
using namespace sim::fem::gpu;

namespace {

struct PipelineFixture : public ::testing::Test {
    std::unique_ptr<Device>             device;
    std::unique_ptr<ShaderCompiler>     compiler;
    std::unique_ptr<GpuElasticHessian>  hess;
    std::unique_ptr<GPUBCOOSorter>      sorter;
    std::unique_ptr<GpuBlockPCGSolver>  solver;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        hess   = std::make_unique<GpuElasticHessian>(*device, *compiler);
        sorter = std::make_unique<GPUBCOOSorter>(*device, *compiler);
        solver = std::make_unique<GpuBlockPCGSolver>(*device, *compiler);
        if (!hess->valid() || !sorter->valid() || !solver->valid())
            GTEST_SKIP() << "GPU pipelines failed to compile";
    }

    BufferRef makeBuf(size_t bytes) {
        return device->createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        });
    }
    void uploadAt(const BufferRef& dst, size_t dstOff, const void* data, size_t bytes) {
        auto st = device->createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
            .usage = BufferDesc::TransferSrc});
        std::memcpy(st->map(), data, bytes); st->unmap();
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, dstOff, bytes}}};
        cmd->copyBuffer(st, dst, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
    }
    void deviceCopy(const BufferRef& src, const BufferRef& dst, size_t bytes) {
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(src, dst, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
    }
    template <class T>
    std::vector<T> download(const BufferRef& src, size_t count) {
        size_t bytes = count * sizeof(T);
        auto rb = device->createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst});
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(src, rb, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
        std::vector<T> out(count);
        std::memcpy(out.data(), rb->map(), bytes);
        rb->unmap();
        return out;
    }
};

}  // namespace

TEST_F(PipelineFixture, ElasticSortSolveMatchesCpu) {
    // Unit cube, 6-tet decomposition sharing the 0-6 diagonal.
    std::vector<glm::dvec3> rest = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    std::vector<std::array<int,4>> tets = {
        {0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}};
    const uint32_t N = uint32_t(rest.size());

    // Current config: small shear so F != I (non-trivial elastic Hessian).
    std::vector<glm::dvec3> cur = rest;
    for (auto& p : cur) { p.x += 0.05 * p.y; p.y += 0.03 * p.z; }

    const double mu = 1.0, lambda = 1.0;
    const double mass = 50.0;   // dominant diagonal -> SPD, well-conditioned

    // ---- 1) GPU elastic Hessian -> device BCOO (E entries) ----
    DeviceBcoo eh = hess->computeToDevice(rest, cur, tets, mu, lambda);
    ASSERT_GT(eh.nnz, 0u);
    const uint32_t E = eh.nnz;
    const uint32_t total = E + N;   // + mass diagonal

    // ---- 2) Combined device BCOO: elastic (front) + mass*I diagonal (tail) ----
    auto bBlocks = makeBuf(size_t(total) * 9 * sizeof(double));
    auto bRow    = makeBuf(size_t(total) * sizeof(uint32_t));
    auto bCol    = makeBuf(size_t(total) * sizeof(uint32_t));
    auto bSeg    = makeBuf(size_t(total + 1) * sizeof(uint32_t));
    deviceCopy(eh.blocks, bBlocks, size_t(E) * 9 * sizeof(double));
    deviceCopy(eh.row,    bRow,    size_t(E) * sizeof(uint32_t));
    deviceCopy(eh.col,    bCol,    size_t(E) * sizeof(uint32_t));

    std::vector<double>   massB(size_t(N) * 9, 0.0);
    std::vector<uint32_t> massR(N), massC(N);
    for (uint32_t i = 0; i < N; ++i) {
        massB[i * 9 + 0] = massB[i * 9 + 4] = massB[i * 9 + 8] = mass; // mass*I
        massR[i] = i; massC[i] = i;
    }
    uploadAt(bBlocks, size_t(E) * 9 * sizeof(double), massB.data(), massB.size() * sizeof(double));
    uploadAt(bRow,    size_t(E) * sizeof(uint32_t),   massR.data(), massR.size() * sizeof(uint32_t));
    uploadAt(bCol,    size_t(E) * sizeof(uint32_t),   massC.data(), massC.size() * sizeof(uint32_t));

    // ---- Reference matrix (download the GPU-assembled BCOO; not in the chain) ----
    auto hostBlocks = download<double>(bBlocks, size_t(total) * 9);
    auto hostRow    = download<uint32_t>(bRow, total);
    auto hostCol    = download<uint32_t>(bCol, total);

    maths::BlockSparseMatrix<3> A(int(N), int(N));
    for (uint32_t k = 0; k < total; ++k) {
        glm::dmat3 b;
        std::memcpy(glm::value_ptr(b), &hostBlocks[k * 9], 9 * sizeof(double));
        A.addBlock(int(hostRow[k]), int(hostCol[k]), b);
    }

    // rhs
    maths::BlockVector<3> bvec(int(N));
    {
        std::mt19937 rng(2024);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (uint32_t i = 0; i < N; ++i) bvec[i] = glm::dvec3(u(rng), u(rng), u(rng));
    }

    // ---- 3) GPU sort + 4) solveDevice (pure device buffers) ----
    uint32_t numSeg = sorter->sort(bBlocks, bRow, bCol, bSeg, total);
    ASSERT_GT(numSeg, 0u);

    auto bRhs = makeBuf(size_t(N) * 3 * sizeof(double));
    auto bX   = makeBuf(size_t(N) * 3 * sizeof(double));
    uploadAt(bRhs, 0, bvec.data(), size_t(N) * 3 * sizeof(double));

    auto gpuRes = solver->solveDevice(bBlocks, bRow, bCol, bSeg, N, total, numSeg,
                                      bRhs, bX, 5000, 1e-12);
    EXPECT_TRUE(gpuRes.converged);

    auto xFlat = download<double>(bX, size_t(N) * 3);

    // ---- CPU reference: same matrix, same rhs ----
    maths::BlockVector<3> xCpu(int(N));
    xCpu.setZero();
    maths::BlockPCGSolver cpu(5000, 1e-12);
    auto cpuRes = cpu.solve(A, bvec, xCpu);
    ASSERT_TRUE(cpuRes.converged);

    double maxErr = 0.0, refMax = 0.0;
    for (uint32_t i = 0; i < N; ++i)
        for (int d = 0; d < 3; ++d) {
            double xg = xFlat[i * 3 + d];
            maxErr = std::max(maxErr, std::abs(xg - xCpu[i][d]));
            refMax = std::max(refMax, std::abs(xCpu[i][d]));
        }
    double relErr = maxErr / std::max(refMax, 1e-30);
    spdlog::info("[test-gpu-fem-pipeline] E={} total={} numSeg={} gpuIters={} relErr={:.3e}",
                 E, total, numSeg, gpuRes.iters, relErr);
    EXPECT_LT(relErr, 1e-6);
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
