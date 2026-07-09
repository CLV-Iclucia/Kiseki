// ============================================================================
// test-gpu-barrier.cc — GpuBarrierAssembler vs CPU constraintPairBarrier*.
//
// Feed identical PP/PE/PT/EE ConstraintPairs (grouped per typeOffsets) + the
// same positions. The GPU per-pair barrier gradient entries and rank-1 Hessian
// blocks, accumulated per vertex / per (row,col), must match the CPU unified
// barrier path. Agreement is to ~1e-7 relative (GPU uses Newton-refined double
// sqrt and a series-based double log; CPU uses std::sqrt/std::log).
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-barrier-assembler.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/gipc/barrier.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <map>
#include <random>
#include <vector>

using namespace sim::rhi;
using namespace sim::fem::gpu;
namespace ipc = sim::fem::ipc;

namespace {

struct BarrierFixture : public ::testing::Test {
    std::unique_ptr<Device>              device;
    std::unique_ptr<ShaderCompiler>      compiler;
    std::unique_ptr<GpuBarrierAssembler> ba;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        ba = std::make_unique<GpuBarrierAssembler>(*device, *compiler);
        if (!ba->valid()) GTEST_SKIP() << "pipeline failed to compile";
    }

    BufferRef makeBuf(size_t bytes) {
        return device->createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst});
    }
    void upload(const BufferRef& dst, const void* data, size_t bytes) {
        auto st = device->createBuffer({.sizeBytes = bytes,
            .visibility = BufferDesc::Visibility::HostVisible, .usage = BufferDesc::TransferSrc});
        std::memcpy(st->map(), data, bytes); st->unmap();
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(st, dst, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
    }
    template <class T>
    std::vector<T> download(const BufferRef& src, size_t count) {
        size_t bytes = count * sizeof(T);
        auto rb = device->createBuffer({.sizeBytes = bytes,
            .visibility = BufferDesc::Visibility::Readback, .usage = BufferDesc::TransferDst});
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(src, rb, rg);
        device->submitAndWait(*cmd, QueueType::Transfer);
        std::vector<T> out(count);
        std::memcpy(out.data(), rb->map(), bytes); rb->unmap();
        return out;
    }
};

}  // namespace

TEST_F(BarrierFixture, MatchesCpuUnifiedBarrier) {
    const uint32_t V = 60;
    const double kappa = 100.0, dHat = 1.0;

    std::mt19937 rng(19);
    std::uniform_real_distribution<double> pos(-0.4, 0.4);
    std::vector<double> x(size_t(V) * 3);
    for (uint32_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) x[i * 3 + d] = pos(rng);

    auto distinct = [&](int* dst, int k) {
        for (int j = 0; j < k; ++j) {
            int v; bool ok;
            do { v = int(rng() % V); ok = true;
                 for (int m = 0; m < j; ++m) if (dst[m] == v) ok = false; } while (!ok);
            dst[j] = v;
        }
    };

    // Build pairs grouped by kind: PP, PE, PT, EE.
    const int nPP = 40, nPE = 40, nPT = 40, nEE = 40;
    std::vector<ipc::ConstraintPair> pairsCpu;
    auto add = [&](ipc::ConstraintKind kind, int nverts, int count) {
        for (int c = 0; c < count; ++c) {
            int q[4] = {-1, -1, -1, -1};
            distinct(q, nverts);
            ipc::ConstraintPair cp;
            cp.type = kind;
            cp.indices = {q[0], q[1], q[2], q[3]};
            pairsCpu.push_back(cp);
        }
    };
    add(ipc::ConstraintKind::PP, 2, nPP);
    add(ipc::ConstraintKind::PE, 3, nPE);
    add(ipc::ConstraintKind::PT, 4, nPT);
    add(ipc::ConstraintKind::EE, 4, nEE);

    const uint32_t total = uint32_t(pairsCpu.size());
    std::array<uint32_t, 5> typeOffsets = {
        0, uint32_t(nPP), uint32_t(nPP + nPE),
        uint32_t(nPP + nPE + nPT), total};

    std::vector<int> pairsFlat(size_t(total) * 4);
    for (uint32_t i = 0; i < total; ++i)
        for (int j = 0; j < 4; ++j) pairsFlat[i * 4 + j] = pairsCpu[i].indices[j];

    // ---- CPU reference ----
    ipc::gipc::Barrier barrier(dHat);
    sim::maths::BlockVector<3> xbv(V), Xbv(V), gradCpu(V);
    for (uint32_t i = 0; i < V; ++i) xbv[i] = glm::dvec3(x[i*3], x[i*3+1], x[i*3+2]);
    gradCpu.setZero();
    sim::maths::BlockSparseMatrix<3> Hcpu;
    for (const auto& cp : pairsCpu) {
        ipc::constraintPairBarrierGradient(cp, xbv, Xbv, gradCpu, barrier, kappa);
        ipc::constraintPairBarrierHessian(cp, xbv, Xbv, Hcpu, barrier, kappa);
    }

    std::map<long long, glm::dmat3> hMapCpu;
    for (size_t k = 0; k < size_t(Hcpu.numEntries()); ++k) {
        long long key = 1LL * Hcpu.rowIndices()[k] * V + Hcpu.colIndices()[k];
        auto it = hMapCpu.find(key);
        if (it == hMapCpu.end()) hMapCpu[key] = Hcpu.blocks()[k];   // avoid glm identity default
        else                     it->second += Hcpu.blocks()[k];
    }

    // ---- GPU ----
    auto bX = makeBuf(size_t(V) * 3 * sizeof(double));
    auto bPairs = makeBuf(size_t(total) * 4 * sizeof(int));
    upload(bX, x.data(), x.size() * sizeof(double));
    upload(bPairs, pairsFlat.data(), pairsFlat.size() * sizeof(int));

    // Rest = 0 (matches the CPU reference's Xbv default) -> eps_x = 0, so the
    // near-parallel mollifier never triggers here; this test pins the plain path.
    std::vector<double> zeros(size_t(V) * 3, 0.0);
    auto bXRest = makeBuf(size_t(V) * 3 * sizeof(double));
    upload(bXRest, zeros.data(), zeros.size() * sizeof(double));

    auto res = ba->assemble(bX, bPairs, typeOffsets, total, kappa, dHat, bXRest);
    ASSERT_GT(res.numHessBlocks, 0u);
    ASSERT_GT(res.numGradEntries, 0u);

    auto gRow = download<uint32_t>(ba->gradRow(), res.numGradEntries);
    auto gVal = download<double>(ba->gradVal(), size_t(res.numGradEntries) * 3);
    auto hRow = download<uint32_t>(ba->hessRow(), res.numHessBlocks);
    auto hCol = download<uint32_t>(ba->hessCol(), res.numHessBlocks);
    auto hBlk = download<double>(ba->hessBlocks(), size_t(res.numHessBlocks) * 9);

    std::vector<glm::dvec3> gradGpu(V, glm::dvec3(0.0));
    for (uint32_t i = 0; i < res.numGradEntries; ++i)
        gradGpu[gRow[i]] += glm::dvec3(gVal[i*3], gVal[i*3+1], gVal[i*3+2]);

    std::map<long long, glm::dmat3> hMapGpu;
    for (uint32_t k = 0; k < res.numHessBlocks; ++k) {
        glm::dmat3 m(0.0);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                m[c][r] = hBlk[size_t(k) * 9 + c * 3 + r];
        long long key = 1LL * hRow[k] * V + hCol[k];
        auto it = hMapGpu.find(key);
        if (it == hMapGpu.end()) hMapGpu[key] = m;
        else                     it->second += m;
    }

    // ---- compare gradient ----
    double maxGradErr = 0.0, gradNorm = 0.0;
    for (uint32_t i = 0; i < V; ++i) {
        for (int d = 0; d < 3; ++d) {
            double a = gradGpu[i][d], b = gradCpu[i][d];
            maxGradErr = std::max(maxGradErr, std::abs(a - b));
            gradNorm = std::max(gradNorm, std::abs(b));
        }
    }
    spdlog::info("[test-gpu-barrier] gradNorm={:.6e} maxGradErr={:.3e}", gradNorm, maxGradErr);
    ASSERT_GT(gradNorm, 0.0);  // some constraints active
    EXPECT_LT(maxGradErr, 1e-6 * std::max(1.0, gradNorm));

    // ---- compare Hessian ----
    EXPECT_EQ(hMapGpu.size(), hMapCpu.size());
    double maxHErr = 0.0, hNorm = 0.0;
    for (const auto& [key, mc] : hMapCpu) {
        auto it = hMapGpu.find(key);
        ASSERT_NE(it, hMapGpu.end()) << "missing block key " << key;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) {
                maxHErr = std::max(maxHErr, std::abs(it->second[c][r] - mc[c][r]));
                hNorm = std::max(hNorm, std::abs(mc[c][r]));
            }
    }
    spdlog::info("[test-gpu-barrier] hNorm={:.6e} maxHErr={:.3e}", hNorm, maxHErr);
    EXPECT_LT(maxHErr, 1e-6 * std::max(1.0, hNorm));
}

// ---- Near-parallel edge-edge: GPU mollified branch vs CPU (both shared) ----
TEST_F(BarrierFixture, MollifiedEEMatchesCpu) {
    const double kappa = 1e3, dHat = 1.0;
    const int nEE = 16;
    const uint32_t V = uint32_t(nEE) * 4;

    // Each EE pair: edge A along +x; edge B near-parallel (tiny skew), offset by
    // h in y, with eb0.y > eb1.y. cross² < eps_x -> mollifier triggers.
    std::vector<double> x(size_t(V) * 3);
    auto setP = [&](int v, glm::dvec3 p) { x[v*3]=p.x; x[v*3+1]=p.y; x[v*3+2]=p.z; };
    const double h = 0.1, dl = 0.01, ep = 0.01;
    for (int i = 0; i < nEE; ++i) {
        double yi = double(i) * 0.5;
        int b = i * 4;
        setP(b + 0, {0.0,  yi,           0.0});
        setP(b + 1, {1.0,  yi,           0.0});
        setP(b + 2, {0.05, yi + h + dl,  0.0});
        setP(b + 3, {1.05, yi + h - dl,  ep });
    }

    const uint32_t total = uint32_t(nEE);
    std::array<uint32_t,5> typeOffsets = {0, 0, 0, 0, total};   // all EE
    std::vector<int> pairsFlat(size_t(total) * 4);
    for (int i = 0; i < nEE; ++i)
        for (int j = 0; j < 4; ++j) pairsFlat[i*4+j] = i*4 + j;

    // ---- CPU reference (rest = current; mollifier active via shared header) ----
    ipc::gipc::Barrier barrier(dHat);
    sim::maths::BlockVector<3> xbv(V), Xbv(V), gradCpu(V);
    for (uint32_t i = 0; i < V; ++i) {
        xbv[i] = glm::dvec3(x[i*3], x[i*3+1], x[i*3+2]);
        Xbv[i] = xbv[i];
    }
    gradCpu.setZero();
    sim::maths::BlockSparseMatrix<3> Hcpu;
    for (int i = 0; i < nEE; ++i) {
        ipc::ConstraintPair cp;
        cp.type = ipc::ConstraintKind::EE;
        cp.indices = {i*4, i*4+1, i*4+2, i*4+3};
        ipc::constraintPairBarrierGradient(cp, xbv, Xbv, gradCpu, barrier, kappa);
        ipc::constraintPairBarrierHessian(cp, xbv, Xbv, Hcpu, barrier, kappa);
    }
    std::map<long long, glm::dmat3> hMapCpu;
    for (size_t k = 0; k < size_t(Hcpu.numEntries()); ++k) {
        long long key = 1LL * Hcpu.rowIndices()[k] * V + Hcpu.colIndices()[k];
        auto it = hMapCpu.find(key);
        if (it == hMapCpu.end()) hMapCpu[key] = Hcpu.blocks()[k];
        else                     it->second += Hcpu.blocks()[k];
    }

    // ---- GPU (rest = current) ----
    auto bX     = makeBuf(size_t(V) * 3 * sizeof(double));
    auto bPairs = makeBuf(size_t(total) * 4 * sizeof(int));
    upload(bX, x.data(), x.size() * sizeof(double));
    upload(bPairs, pairsFlat.data(), pairsFlat.size() * sizeof(int));
    auto res = ba->assemble(bX, bPairs, typeOffsets, total, kappa, dHat, bX);
    ASSERT_GT(res.numHessBlocks, 0u);
    ASSERT_GT(res.numGradEntries, 0u);

    auto gRow = download<uint32_t>(ba->gradRow(), res.numGradEntries);
    auto gVal = download<double>(ba->gradVal(), size_t(res.numGradEntries) * 3);
    auto hRow = download<uint32_t>(ba->hessRow(), res.numHessBlocks);
    auto hCol = download<uint32_t>(ba->hessCol(), res.numHessBlocks);
    auto hBlk = download<double>(ba->hessBlocks(), size_t(res.numHessBlocks) * 9);

    std::vector<glm::dvec3> gradGpu(V, glm::dvec3(0.0));
    for (uint32_t i = 0; i < res.numGradEntries; ++i)
        gradGpu[gRow[i]] += glm::dvec3(gVal[i*3], gVal[i*3+1], gVal[i*3+2]);

    std::map<long long, glm::dmat3> hMapGpu;
    for (uint32_t k = 0; k < res.numHessBlocks; ++k) {
        glm::dmat3 m(0.0);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                m[c][r] = hBlk[size_t(k) * 9 + c * 3 + r];
        long long key = 1LL * hRow[k] * V + hCol[k];
        auto it = hMapGpu.find(key);
        if (it == hMapGpu.end()) hMapGpu[key] = m;
        else                     it->second += m;
    }

    // ---- compare gradient ----
    double maxGradErr = 0.0, gradNorm = 0.0;
    for (uint32_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) {
            maxGradErr = std::max(maxGradErr, std::abs(gradGpu[i][d] - gradCpu[i][d]));
            gradNorm = std::max(gradNorm, std::abs(gradCpu[i][d]));
        }
    spdlog::info("[test-gpu-barrier/molli] gradNorm={:.6e} maxGradErr={:.3e}", gradNorm, maxGradErr);
    ASSERT_GT(gradNorm, 0.0);   // mollified contributions are non-zero
    EXPECT_LT(maxGradErr, 1e-6 * std::max(1.0, gradNorm));

    // ---- compare Hessian ----
    double maxHErr = 0.0, hNorm = 0.0;
    for (const auto& [key, mc] : hMapCpu) {
        auto it = hMapGpu.find(key);
        ASSERT_NE(it, hMapGpu.end()) << "missing GPU block key " << key;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) {
                maxHErr = std::max(maxHErr, std::abs(it->second[c][r] - mc[c][r]));
                hNorm = std::max(hNorm, std::abs(mc[c][r]));
            }
    }
    spdlog::info("[test-gpu-barrier/molli] hNorm={:.6e} maxHErr={:.3e}", hNorm, maxHErr);
    ASSERT_GT(hNorm, 0.0);
    EXPECT_LT(maxHErr, 1e-6 * std::max(1.0, hNorm));
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
