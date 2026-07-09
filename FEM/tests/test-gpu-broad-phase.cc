// ============================================================================
// test-gpu-broad-phase.cc — GpuBroadPhase VT/EE candidates vs brute force.
//
// The LBVH query must return exactly the pairs that pass the same box-overlap +
// skip test as an O(V*T) / O(E*E) brute-force scan (the BVH only accelerates it).
// Query AABB = primitive trajectory AABB dilated by dHat; leaf AABB = raw. EE is
// NOT i<j deduped (each edge queries all edges), matching the CPU integrator.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-trajectory-bounds.h>
#include <fem/gpu/gpu-lbvh.h>
#include <fem/gpu/gpu-broad-phase.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace sim::rhi;
using namespace sim::fem::gpu;

namespace {

struct BroadPhaseFixture : public ::testing::Test {
    std::unique_ptr<Device>              device;
    std::unique_ptr<ShaderCompiler>      compiler;
    std::unique_ptr<GpuTrajectoryBounds> traj;
    std::unique_ptr<GPULBVH>             triBvh, edgeBvh;
    std::unique_ptr<GpuBroadPhase>       bp;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        traj    = std::make_unique<GpuTrajectoryBounds>(*device);
        triBvh  = std::make_unique<GPULBVH>(*device, *compiler);
        edgeBvh = std::make_unique<GPULBVH>(*device, *compiler);
        bp      = std::make_unique<GpuBroadPhase>(*device, *compiler);
        if (!traj->valid() || !triBvh->valid() || !edgeBvh->valid() || !bp->valid())
            GTEST_SKIP() << "pipelines failed to compile";
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

}  // namespace

TEST_F(BroadPhaseFixture, VtEeMatchBruteForce) {
    const uint32_t V = 150, T = 250, E = 250;
    const double alpha = 0.6, dHat = 0.06;

    std::mt19937 rng(2025);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    std::uniform_real_distribution<double> dir(-0.5, 0.5);
    std::uniform_int_distribution<uint32_t> vpick(0, V - 1);

    std::vector<double> x(size_t(V) * 3), p(size_t(V) * 3);
    for (uint32_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) { x[i * 3 + d] = pos(rng); p[i * 3 + d] = dir(rng); }

    std::vector<uint32_t> triConn(size_t(T) * 3), edgeConn(size_t(E) * 2), vertConn(V);
    for (uint32_t t = 0; t < T; ++t) for (int j = 0; j < 3; ++j) triConn[t * 3 + j] = vpick(rng);
    for (uint32_t e = 0; e < E; ++e) for (int j = 0; j < 2; ++j) edgeConn[e * 2 + j] = vpick(rng);
    for (uint32_t v = 0; v < V; ++v) vertConn[v] = v;

    // ---- helpers ----
    auto vtraj = [&](uint32_t v, glm::dvec3& lo, glm::dvec3& hi) {
        glm::dvec3 xs(x[v*3], x[v*3+1], x[v*3+2]);
        glm::dvec3 xe = xs + alpha * glm::dvec3(p[v*3], p[v*3+1], p[v*3+2]);
        lo = glm::min(xs, xe); hi = glm::max(xs, xe);
    };
    auto primTraj = [&](const uint32_t* vs, int k, glm::dvec3& lo, glm::dvec3& hi) {
        lo = glm::dvec3(1e300); hi = glm::dvec3(-1e300);
        for (int j = 0; j < k; ++j) {
            glm::dvec3 l, h; vtraj(vs[j], l, h);
            lo = glm::min(lo, l); hi = glm::max(hi, h);
        }
    };
    auto overlap = [](glm::dvec3 alo, glm::dvec3 ahi, glm::dvec3 blo, glm::dvec3 bhi) {
        for (int d = 0; d < 3; ++d) if (alo[d] > bhi[d] || ahi[d] < blo[d]) return false;
        return true;
    };

    // ---- brute-force reference ----
    std::vector<std::array<uint32_t, 4>> refVT, refEE;
    for (uint32_t v = 0; v < V; ++v) {
        glm::dvec3 vl, vh; vtraj(v, vl, vh);
        vl -= dHat; vh += dHat;
        for (uint32_t t = 0; t < T; ++t) {
            glm::dvec3 tl, th; primTraj(&triConn[t * 3], 3, tl, th);
            uint32_t a = triConn[t*3], b = triConn[t*3+1], c = triConn[t*3+2];
            if (a == v || b == v || c == v) continue;
            if (overlap(vl, vh, tl, th)) refVT.push_back({v, a, b, c});
        }
    }
    for (uint32_t i = 0; i < E; ++i) {
        glm::dvec3 il, ih; primTraj(&edgeConn[i * 2], 2, il, ih);
        il -= dHat; ih += dHat;
        uint32_t a0 = edgeConn[i*2], a1 = edgeConn[i*2+1];
        for (uint32_t j = 0; j < E; ++j) {
            glm::dvec3 jl, jh; primTraj(&edgeConn[j * 2], 2, jl, jh);
            uint32_t b0 = edgeConn[j*2], b1 = edgeConn[j*2+1];
            if (a0==b0||a0==b1||a1==b0||a1==b1) continue;
            if (overlap(il, ih, jl, jh)) refEE.push_back({a0, a1, b0, b1});
        }
    }

    // ---- GPU pipeline ----
    auto bX = makeBuf(size_t(V)*3*sizeof(double)), bP = makeBuf(size_t(V)*3*sizeof(double));
    auto bTriConn = makeBuf(size_t(T)*3*sizeof(uint32_t));
    auto bEdgeConn= makeBuf(size_t(E)*2*sizeof(uint32_t));
    auto bVertConn= makeBuf(size_t(V)*sizeof(uint32_t));
    auto bAlpha = makeBuf(sizeof(double)), bDHat = makeBuf(sizeof(double));
    upload(bX, x.data(), x.size()*sizeof(double));
    upload(bP, p.data(), p.size()*sizeof(double));
    upload(bTriConn, triConn.data(), triConn.size()*sizeof(uint32_t));
    upload(bEdgeConn, edgeConn.data(), edgeConn.size()*sizeof(uint32_t));
    upload(bVertConn, vertConn.data(), vertConn.size()*sizeof(uint32_t));
    upload(bAlpha, &alpha, sizeof(double));
    upload(bDHat, &dHat, sizeof(double));

    auto triLo = makeBuf(size_t(T)*3*sizeof(double)), triHi = makeBuf(size_t(T)*3*sizeof(double));
    auto edgeLo= makeBuf(size_t(E)*3*sizeof(double)), edgeHi= makeBuf(size_t(E)*3*sizeof(double));
    auto vertLo= makeBuf(size_t(V)*3*sizeof(double)), vertHi= makeBuf(size_t(V)*3*sizeof(double));

    traj->compute(bX, bP, bTriConn,  bAlpha, triLo,  triHi,  T, 3);
    traj->compute(bX, bP, bEdgeConn, bAlpha, edgeLo, edgeHi, E, 2);
    traj->compute(bX, bP, bVertConn, bAlpha, vertLo, vertHi, V, 1);

    triBvh->build(triLo, triHi, T);
    edgeBvh->build(edgeLo, edgeHi, E);

    uint32_t vtCount = bp->queryVT(*triBvh, vertLo, vertHi, bTriConn, bDHat, V);
    uint32_t eeCount = bp->queryEE(*edgeBvh, edgeLo, edgeHi, bEdgeConn, bDHat, E);

    auto vtFlat = vtCount ? download<uint32_t>(bp->vtPairs(), size_t(vtCount) * 4)
                          : std::vector<uint32_t>{};
    auto eeFlat = eeCount ? download<uint32_t>(bp->eePairs(), size_t(eeCount) * 4)
                          : std::vector<uint32_t>{};

    std::vector<std::array<uint32_t, 4>> gotVT(vtCount), gotEE(eeCount);
    for (uint32_t k = 0; k < vtCount; ++k)
        gotVT[k] = {vtFlat[k*4], vtFlat[k*4+1], vtFlat[k*4+2], vtFlat[k*4+3]};
    for (uint32_t k = 0; k < eeCount; ++k)
        gotEE[k] = {eeFlat[k*4], eeFlat[k*4+1], eeFlat[k*4+2], eeFlat[k*4+3]};

    std::sort(refVT.begin(), refVT.end());  std::sort(gotVT.begin(), gotVT.end());
    std::sort(refEE.begin(), refEE.end());  std::sort(gotEE.begin(), gotEE.end());

    spdlog::info("[test-gpu-broad-phase] VT gpu={} ref={} ; EE gpu={} ref={}",
                 gotVT.size(), refVT.size(), gotEE.size(), refEE.size());
    EXPECT_EQ(gotVT, refVT);
    EXPECT_EQ(gotEE, refEE);
    ASSERT_GT(refVT.size(), 0u);  // sanity: scene actually produces candidates
    ASSERT_GT(refEE.size(), 0u);
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
