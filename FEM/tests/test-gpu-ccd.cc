// ============================================================================
// test-gpu-ccd.cc — GpuCcd additive CCD vs the shared CPU ACCD.
//
// For arbitrary VT/EE candidate lists + positions + a Newton step direction,
// the GPU must compute the same per-candidate time-of-impact as the shared
// ACCD (the single source behind CollisionDetector::runACCD) and reduce them to
// the same conservative step-size upper bound. The GPU uses a Newton-refined
// double sqrt (no native double sqrt intrinsic), so agreement is to ~1e-9
// relative rather than bit-exact.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-ccd.h>
#include <fem/shared/ipc-ccd.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace ksk::rhi;
using namespace ksk::fem::gpu;
namespace shared = ksk::fem::shared;

namespace {

struct CcdFixture : public ::testing::Test {
    std::unique_ptr<Device>         device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<GPUACCD>         ccd;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        ccd = std::make_unique<GPUACCD>(*device, *compiler);
        if (!ccd->valid()) GTEST_SKIP() << "pipelines failed to compile";
    }

    BufferRef makeBuf(size_t bytes) {
        return device->createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        });
    }
    void upload(const BufferRef& dst, const void* data, size_t bytes) {
        auto st = device->createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
            .usage = BufferDesc::TransferSrc});
        std::memcpy(st->map(), data, bytes); st->unmap();
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(st, dst, rg);
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

glm::dvec3 vget(const std::vector<double>& b, int i) {
    return {b[i * 3 + 0], b[i * 3 + 1], b[i * 3 + 2]};
}

}  // namespace

TEST_F(CcdFixture, MatchesSharedAccdAndStepBound) {
    const uint32_t V = 80, Nvt = 600, Nee = 600;
    const double toiCap = 1.0, s = 0.1;

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    std::uniform_real_distribution<double> dir(-0.6, 0.6);

    std::vector<double> x(size_t(V) * 3), p(size_t(V) * 3);
    for (uint32_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) { x[i*3+d] = pos(rng); p[i*3+d] = dir(rng); }

    auto distinct = [&](int* dst, int k) {
        for (int j = 0; j < k; ++j) {
            int v; bool ok;
            do { v = int(rng() % V); ok = true;
                 for (int m = 0; m < j; ++m) if (dst[m] == v) ok = false; } while (!ok);
            dst[j] = v;
        }
    };

    std::vector<uint32_t> vtCand(size_t(Nvt) * 4), eeCand(size_t(Nee) * 4);
    for (uint32_t i = 0; i < Nvt; ++i) {
        int q[4]; distinct(q, 4);
        for (int j = 0; j < 4; ++j) vtCand[i*4+j] = uint32_t(q[j]);
    }
    for (uint32_t i = 0; i < Nee; ++i) {
        int q[4]; distinct(q, 4);
        for (int j = 0; j < 4; ++j) eeCand[i*4+j] = uint32_t(q[j]);
    }

    // ---- CPU reference via shared ACCD (single source behind runACCD) ----
    const uint32_t Nc = Nvt + Nee;
    std::vector<double> cpuToi(Nc);
    double cpuMin = toiCap;
    for (uint32_t i = 0; i < Nvt; ++i) {
        int v=int(vtCand[i*4]), t0=int(vtCand[i*4+1]), t1=int(vtCand[i*4+2]), t2=int(vtCand[i*4+3]);
        double r = shared::shACCD(SH_CCD_VT,
            vget(x,v), vget(x,t0), vget(x,t1), vget(x,t2),
            vget(p,v), vget(p,t0), vget(p,t1), vget(p,t2), toiCap, s);
        cpuToi[i] = r; cpuMin = std::min(cpuMin, r);
    }
    for (uint32_t i = 0; i < Nee; ++i) {
        int a0=int(eeCand[i*4]), a1=int(eeCand[i*4+1]), b0=int(eeCand[i*4+2]), b1=int(eeCand[i*4+3]);
        double r = shared::shACCD(SH_CCD_EE,
            vget(x,a0), vget(x,a1), vget(x,b0), vget(x,b1),
            vget(p,a0), vget(p,a1), vget(p,b0), vget(p,b1), toiCap, s);
        cpuToi[Nvt + i] = r; cpuMin = std::min(cpuMin, r);
    }

    // ---- GPU ----
    auto bX  = makeBuf(size_t(V)*3*sizeof(double));
    auto bP  = makeBuf(size_t(V)*3*sizeof(double));
    auto bVt = makeBuf(size_t(Nvt)*4*sizeof(uint32_t));
    auto bEe = makeBuf(size_t(Nee)*4*sizeof(uint32_t));
    upload(bX, x.data(), x.size()*sizeof(double));
    upload(bP, p.data(), p.size()*sizeof(double));
    upload(bVt, vtCand.data(), vtCand.size()*sizeof(uint32_t));
    upload(bEe, eeCand.data(), eeCand.size()*sizeof(uint32_t));

    double gpuStep = ccd->stepSizeUpperBound(bX, bP, bVt, Nvt, bEe, Nee, toiCap, s);
    auto gpuToi = download<double>(ccd->tois(), Nc);

    spdlog::info("[test-gpu-ccd] step gpu={:.12f} cpu={:.12f}", gpuStep, cpuMin);

    // per-candidate agreement (Newton-sqrt vs std::sqrt -> ~1e-9 relative)
    uint32_t mism = 0;
    for (uint32_t e = 0; e < Nc; ++e) {
        double tol = 1e-7 * std::max(1.0, std::abs(cpuToi[e]));
        if (std::abs(gpuToi[e] - cpuToi[e]) > tol) ++mism;
    }
    EXPECT_EQ(mism, 0u) << mism << " / " << Nc << " per-candidate toi mismatches";

    EXPECT_NEAR(gpuStep, cpuMin, 1e-7 * std::max(1.0, std::abs(cpuMin)));
    ASSERT_LT(cpuMin, toiCap);  // sanity: scene actually limits the step
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
