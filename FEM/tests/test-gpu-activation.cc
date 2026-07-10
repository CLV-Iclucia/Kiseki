// ============================================================================
// test-gpu-activation.cc — GpuActivation vs CPU refreshActiveConstraintPairs.
//
// Feed arbitrary VT/EE candidate lists + positions; the GPU must classify each
// (point-triangle / edge-edge distance type), filter by dHat^2, remap to the
// unified ConstraintPair (kind + indices), and bucket into PP/PE/PT/EE with
// compact typeOffsets — byte-identical to the CPU path. Because all the math is
// the SHARED single source (ipc-distance / ipc-activation), this also verifies
// HLSL compiles that logic and computes the same doubles (no float downcast).
//
// Comparison is per-kind multiset of 4-tuples (within-bucket order is a stable-
// sort detail that does not affect the assembled barrier).
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-activation.h>
#include <fem/ipc/distances.h>
#include <fem/shared/ipc-activation.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace ksk::rhi;
using namespace ksk::fem::gpu;
namespace ipc    = ksk::fem::ipc;
namespace shared = ksk::fem::shared;

namespace {

struct ActivationFixture : public ::testing::Test {
    std::unique_ptr<Device>         device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<GpuActivation>  act;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        act = std::make_unique<GpuActivation>(*device, *compiler);
        if (!act->valid()) GTEST_SKIP() << "pipelines failed to compile";
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

glm::dvec3 vpos(const std::vector<double>& x, int i) {
    return {x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]};
}

}  // namespace

TEST_F(ActivationFixture, MatchesCpuRefreshActiveConstraintPairs) {
    const uint32_t V = 80, Nvt = 500, Nee = 500;
    const double dHat = 0.7, dHatSqr = dHat * dHat;

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);

    std::vector<double> x(size_t(V) * 3);
    for (uint32_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) x[i * 3 + d] = pos(rng);

    // distinct random vertex indices
    auto distinct = [&](int* dst, int k) {
        for (int j = 0; j < k; ++j) {
            int v;
            bool ok;
            do {
                v = int(rng() % V);
                ok = true;
                for (int m = 0; m < j; ++m) if (dst[m] == v) ok = false;
            } while (!ok);
            dst[j] = v;
        }
    };

    std::vector<uint32_t> vtCand(size_t(Nvt) * 4), eeCand(size_t(Nee) * 4);
    for (uint32_t i = 0; i < Nvt; ++i) {
        int q[4]; distinct(q, 4);  // v, t0, t1, t2 all distinct
        for (int j = 0; j < 4; ++j) vtCand[i * 4 + j] = uint32_t(q[j]);
    }
    for (uint32_t i = 0; i < Nee; ++i) {
        int q[4]; distinct(q, 4);  // a0,a1,b0,b1 all distinct (non-adjacent edges)
        for (int j = 0; j < 4; ++j) eeCand[i * 4 + j] = uint32_t(q[j]);
    }

    // ---- CPU reference (shared single-source logic) bucketed by kind ----
    std::array<std::vector<std::array<int, 4>>, 4> expByKind;
    auto record = [&](const shared::ShConstraintPair& cp) {
        if (cp.kind < 0 || cp.kind > 3) return;
        expByKind[cp.kind].push_back(
            {cp.indices[0], cp.indices[1], cp.indices[2], cp.indices[3]});
    };
    for (uint32_t i = 0; i < Nvt; ++i) {
        int v = int(vtCand[i*4]), t0 = int(vtCand[i*4+1]),
            t1 = int(vtCand[i*4+2]), t2 = int(vtCand[i*4+3]);
        glm::dvec3 p = vpos(x, v), a = vpos(x, t0), b = vpos(x, t1), c = vpos(x, t2);
        auto   type = ipc::decidePointTriangleDistanceType(p, a, b, c);
        double d2   = ipc::distanceSqrPointTriangle(p, a, b, c);
        if (d2 < dHatSqr) record(shared::shActivateVT(int(type), v, t0, t1, t2));
    }
    for (uint32_t i = 0; i < Nee; ++i) {
        int a0 = int(eeCand[i*4]), a1 = int(eeCand[i*4+1]),
            b0 = int(eeCand[i*4+2]), b1 = int(eeCand[i*4+3]);
        glm::dvec3 ea0 = vpos(x, a0), ea1 = vpos(x, a1),
                   eb0 = vpos(x, b0), eb1 = vpos(x, b1);
        auto   type = ipc::decideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1);
        double d2   = ipc::distanceSqrEdgeEdge(ea0, ea1, eb0, eb1);
        if (d2 < dHatSqr) record(shared::shActivateEE(int(type), a0, a1, b0, b1));
    }

    // ---- GPU ----
    auto bX  = makeBuf(size_t(V) * 3 * sizeof(double));
    auto bVt = makeBuf(size_t(Nvt) * 4 * sizeof(uint32_t));
    auto bEe = makeBuf(size_t(Nee) * 4 * sizeof(uint32_t));
    auto bDHatSqr = makeBuf(sizeof(double));
    upload(bX, x.data(), x.size() * sizeof(double));
    upload(bVt, vtCand.data(), vtCand.size() * sizeof(uint32_t));
    upload(bEe, eeCand.data(), eeCand.size() * sizeof(uint32_t));
    upload(bDHatSqr, &dHatSqr, sizeof(double));

    auto res = act->activate(bX, bVt, Nvt, bEe, Nee, bDHatSqr);

    auto flat = res.numConstraints
                    ? download<int>(act->pairs(), size_t(res.numConstraints) * 4)
                    : std::vector<int>{};

    // ---- compare per kind ----
    const uint32_t expTotal = uint32_t(expByKind[0].size() + expByKind[1].size() +
                                       expByKind[2].size() + expByKind[3].size());
    spdlog::info("[test-gpu-activation] total gpu={} cpu={} | PP={}/{} PE={}/{} PT={}/{} EE={}/{}",
                 res.numConstraints, expTotal,
                 res.typeOffsets[1] - res.typeOffsets[0], expByKind[0].size(),
                 res.typeOffsets[2] - res.typeOffsets[1], expByKind[1].size(),
                 res.typeOffsets[3] - res.typeOffsets[2], expByKind[2].size(),
                 res.typeOffsets[4] - res.typeOffsets[3], expByKind[3].size());

    ASSERT_EQ(res.numConstraints, expTotal);
    ASSERT_GT(expTotal, 0u);

    for (int k = 0; k < 4; ++k) {
        uint32_t lo = res.typeOffsets[k], hi = res.typeOffsets[k + 1];
        std::vector<std::array<int, 4>> got;
        for (uint32_t s = lo; s < hi; ++s)
            got.push_back({flat[s*4], flat[s*4+1], flat[s*4+2], flat[s*4+3]});

        auto exp = expByKind[k];
        EXPECT_EQ(got.size(), exp.size()) << "kind " << k << " count";
        std::sort(got.begin(), got.end());
        std::sort(exp.begin(), exp.end());
        EXPECT_EQ(got, exp) << "kind " << k << " constraint set mismatch";
    }
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
