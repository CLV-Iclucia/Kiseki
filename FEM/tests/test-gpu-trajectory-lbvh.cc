// ============================================================================
// test-gpu-trajectory-lbvh.cc — module A: trajectory AABB + GPU scene-bound
// reduction + device-resident GpuLBVH build.
//
// Checks, against CPU references built from the identical inputs:
//   * per-primitive trajectory AABB == union over verts of {x_v, x_v+alpha*p_v}
//   * GPU scene-bound reduction == serial union of all AABBs (bit-exact)
//   * GpuLBVH (with internal scene bound) node AABBs + leaf order == spatify::LBVH
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-trajectory-bounds.h>
#include <fem/gpu/gpu-lbvh.h>
#include <Spatify/lbvh.h>
#include <Spatify/bbox.h>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace ksk::rhi;
using namespace ksk::fem::gpu;

namespace {

using Box = spatify::BBox<double, 3>;

struct Accessor {
    using CoordType = double;
    const std::vector<Box>* boxes;
    [[nodiscard]] int size() const { return int(boxes->size()); }
    [[nodiscard]] Box bbox(int i) const { return (*boxes)[i]; }
};

struct TrajLbvhFixture : public ::testing::Test {
    std::unique_ptr<Device>              device;
    std::unique_ptr<ShaderCompiler>      compiler;
    std::unique_ptr<GpuTrajectoryBounds> traj;
    std::unique_ptr<GPULBVH>             bvh;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        traj = std::make_unique<GpuTrajectoryBounds>(*device);
        bvh  = std::make_unique<GPULBVH>(*device, *compiler);
        if (!traj->valid() || !bvh->valid()) GTEST_SKIP() << "pipelines failed to compile";
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

    void runCase(uint32_t nVerts, uint32_t M, double alpha, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> pos(-10.0, 10.0);
        std::uniform_real_distribution<double> dir(-1.0, 1.0);
        std::uniform_int_distribution<uint32_t> vpick(0, nVerts - 1);

        std::vector<double> x(size_t(nVerts) * 3), p(size_t(nVerts) * 3);
        for (uint32_t i = 0; i < nVerts; ++i)
            for (int d = 0; d < 3; ++d) {
                x[i * 3 + d] = pos(rng);
                p[i * 3 + d] = dir(rng);
            }
        std::vector<uint32_t> conn(size_t(M) * 3);
        for (uint32_t t = 0; t < M; ++t)
            for (int j = 0; j < 3; ++j) conn[t * 3 + j] = vpick(rng);

        // ---- CPU trajectory boxes ----
        std::vector<Box> boxes(M);
        for (uint32_t t = 0; t < M; ++t) {
            Box b;
            for (int j = 0; j < 3; ++j) {
                uint32_t v = conn[t * 3 + j];
                glm::dvec3 xs(x[v * 3], x[v * 3 + 1], x[v * 3 + 2]);
                glm::dvec3 xe = xs + alpha * glm::dvec3(p[v * 3], p[v * 3 + 1], p[v * 3 + 2]);
                b.expand(xs); b.expand(xe);
            }
            boxes[t] = b;
        }
        double sb[6] = { 1e300, 1e300, 1e300, -1e300, -1e300, -1e300 };
        for (uint32_t t = 0; t < M; ++t)
            for (int d = 0; d < 3; ++d) {
                sb[d]     = std::min(sb[d], boxes[t].lo[d]);
                sb[3 + d] = std::max(sb[3 + d], boxes[t].hi[d]);
            }

        spatify::LBVH<double> cpu;
        cpu.update(Accessor{&boxes});
        const uint32_t numNodes = 2 * M - 1;

        // ---- GPU ----
        auto bX     = makeBuf(size_t(nVerts) * 3 * sizeof(double));
        auto bP     = makeBuf(size_t(nVerts) * 3 * sizeof(double));
        auto bConn  = makeBuf(size_t(M) * 3 * sizeof(uint32_t));
        auto bAlpha = makeBuf(sizeof(double));
        auto bLo    = makeBuf(size_t(M) * 3 * sizeof(double));
        auto bHi    = makeBuf(size_t(M) * 3 * sizeof(double));
        upload(bX, x.data(), x.size() * sizeof(double));
        upload(bP, p.data(), p.size() * sizeof(double));
        upload(bConn, conn.data(), conn.size() * sizeof(uint32_t));
        upload(bAlpha, &alpha, sizeof(double));

        traj->compute(bX, bP, bConn, bAlpha, bLo, bHi, M, 3);
        bvh->build(bLo, bHi, M);   // internal GPU scene bound

        auto gLo  = download<double>(bLo, size_t(M) * 3);
        auto gHi  = download<double>(bHi, size_t(M) * 3);
        auto gSB  = download<double>(bvh->sceneBound(), 6);
        auto nLo  = download<double>(bvh->nodeLo(), size_t(numNodes) * 3);
        auto nHi  = download<double>(bvh->nodeHi(), size_t(numNodes) * 3);
        auto gIdx = download<uint32_t>(bvh->sortedIdx(), M);

        // 1) trajectory AABB bit-exact
        for (uint32_t t = 0; t < M; ++t)
            for (int d = 0; d < 3; ++d) {
                EXPECT_EQ(gLo[t * 3 + d], boxes[t].lo[d]) << "prim " << t << " lo " << d;
                EXPECT_EQ(gHi[t * 3 + d], boxes[t].hi[d]) << "prim " << t << " hi " << d;
            }
        // 2) scene bound bit-exact
        for (int d = 0; d < 6; ++d) EXPECT_EQ(gSB[d], sb[d]) << "sceneBound " << d;
        // 3) LBVH node AABBs + leaf order
        for (uint32_t i = 0; i < M; ++i)
            ASSERT_EQ(gIdx[i], uint32_t(cpu.primitiveIndex(int(M - 1 + i)))) << "leaf " << i;
        for (uint32_t k = 0; k < numNodes; ++k)
            for (int d = 0; d < 3; ++d) {
                EXPECT_EQ(nLo[k * 3 + d], cpu.bbox[k].lo[d]) << "node " << k << " lo " << d;
                EXPECT_EQ(nHi[k * 3 + d], cpu.bbox[k].hi[d]) << "node " << k << " hi " << d;
            }

        spdlog::info("[test-gpu-trajectory-lbvh] nVerts={} M={} alpha={} OK", nVerts, M, alpha);
    }
};

}  // namespace

TEST_F(TrajLbvhFixture, StaticSmall)  { runCase(50,   64,   0.0, 1); }
TEST_F(TrajLbvhFixture, MovingMedium) { runCase(500,  2000, 0.7, 2); }
TEST_F(TrajLbvhFixture, MovingLarge)  { runCase(3000, 20000,1.0, 3); }

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
