// ============================================================================
// test-gpu-lbvh.cc — GpuLBVH vs spatify::LBVH<double>.
//
// Builds a BVH over random AABBs on the GPU and checks, against the reference
// CPU LBVH built from the identical input:
//   * leaf primitive order (sortedIdx) is identical (same Morton + stable sort),
//   * every node AABB matches bit-for-bit (same topology + exact refit),
//   * a set of random box-overlap queries returns exactly the brute-force set
//     when traversing the downloaded GPU BVH (functional correctness).
// ============================================================================
#ifdef FEM_GPU_ENABLED

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
    [[nodiscard]] int  size() const { return int(boxes->size()); }
    [[nodiscard]] Box  bbox(int i) const { return (*boxes)[i]; }
};

struct LbvhFixture : public ::testing::Test {
    std::unique_ptr<Device>         device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<GPULBVH>        bvh;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        bvh = std::make_unique<GPULBVH>(*device, *compiler);
        if (!bvh->valid()) GTEST_SKIP() << "LBVH pipelines failed to compile";
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

    void runCase(uint32_t N, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> pos(-10.0, 10.0);
        std::uniform_real_distribution<double> ext(0.05, 0.8);

        std::vector<Box> boxes(N);
        std::vector<double> lo(size_t(N) * 3), hi(size_t(N) * 3);
        for (uint32_t i = 0; i < N; ++i) {
            glm::dvec3 c(pos(rng), pos(rng), pos(rng));
            glm::dvec3 e(ext(rng), ext(rng), ext(rng));
            glm::dvec3 l = c - e, h = c + e;
            boxes[i] = Box(l, h);
            for (int d = 0; d < 3; ++d) { lo[i * 3 + d] = l[d]; hi[i * 3 + d] = h[d]; }
        }

        // Scene bound = union of all AABBs (exact min/max, matches CPU sceneBound).
        double sb[6] = { 1e300, 1e300, 1e300, -1e300, -1e300, -1e300 };
        for (uint32_t i = 0; i < N; ++i)
            for (int d = 0; d < 3; ++d) {
                sb[d]     = std::min(sb[d], lo[i * 3 + d]);
                sb[3 + d] = std::max(sb[3 + d], hi[i * 3 + d]);
            }

        // ---- CPU reference ----
        spatify::LBVH<double> cpu;
        cpu.update(Accessor{&boxes});
        const uint32_t numNodes = 2 * N - 1;

        // ---- GPU build ----
        auto bLo = makeBuf(size_t(N) * 3 * sizeof(double));
        auto bHi = makeBuf(size_t(N) * 3 * sizeof(double));
        auto bSB = makeBuf(6 * sizeof(double));
        upload(bLo, lo.data(), lo.size() * sizeof(double));
        upload(bHi, hi.data(), hi.size() * sizeof(double));
        upload(bSB, sb, sizeof(sb));

        bvh->build(bLo, bHi, bSB, N);

        auto gLo  = download<double>(bvh->nodeLo(), size_t(numNodes) * 3);
        auto gHi  = download<double>(bvh->nodeHi(), size_t(numNodes) * 3);
        auto gIdx = download<uint32_t>(bvh->sortedIdx(), N);
        auto gLch = download<int32_t>(bvh->leftChild(), N - 1);
        auto gRch = download<int32_t>(bvh->rightChild(), N - 1);

        // ---- 1) leaf primitive order ----
        for (uint32_t i = 0; i < N; ++i)
            ASSERT_EQ(gIdx[i], uint32_t(cpu.primitiveIndex(int(N - 1 + i)))) << "leaf " << i;

        // ---- 2) every node AABB matches bit-for-bit ----
        for (uint32_t k = 0; k < numNodes; ++k)
            for (int d = 0; d < 3; ++d) {
                EXPECT_EQ(gLo[k * 3 + d], cpu.bbox[k].lo[d]) << "node " << k << " lo " << d;
                EXPECT_EQ(gHi[k * 3 + d], cpu.bbox[k].hi[d]) << "node " << k << " hi " << d;
            }

        // ---- 3) functional box-overlap queries (GPU BVH vs brute force) ----
        auto overlap = [](const double* alo, const double* ahi,
                          const glm::dvec3& qlo, const glm::dvec3& qhi) {
            for (int d = 0; d < 3; ++d)
                if (alo[d] > qhi[d] || ahi[d] < qlo[d]) return false;
            return true;
        };
        std::mt19937 qrng(seed ^ 0xABCDu);
        for (int q = 0; q < 20; ++q) {
            glm::dvec3 c(pos(qrng), pos(qrng), pos(qrng));
            glm::dvec3 e(1.0, 1.0, 1.0);
            glm::dvec3 qlo = c - e, qhi = c + e;

            std::vector<uint32_t> ref;
            for (uint32_t i = 0; i < N; ++i)
                if (overlap(&lo[i * 3], &hi[i * 3], qlo, qhi)) ref.push_back(i);

            std::vector<uint32_t> got;
            std::array<int, 64> stack{};
            int top = 0, node = 0;
            while (true) {
                bool hit = overlap(&gLo[node * 3], &gHi[node * 3], qlo, qhi);
                if (hit) {
                    if (node >= int(N - 1)) {                 // leaf
                        got.push_back(gIdx[node - int(N - 1)]);
                    } else {
                        stack[top++] = gRch[node];
                        node = gLch[node];
                        continue;
                    }
                }
                if (top == 0) break;
                node = stack[--top];
            }
            std::sort(ref.begin(), ref.end());
            std::sort(got.begin(), got.end());
            ASSERT_EQ(got, ref) << "query " << q;
        }

        spdlog::info("[test-gpu-lbvh] N={} nodes={} OK (bbox bit-exact, queries match)",
                     N, numNodes);
    }
};

}  // namespace

TEST_F(LbvhFixture, Small)   { runCase(64,    1); }
TEST_F(LbvhFixture, Medium)  { runCase(2000,  2); }
TEST_F(LbvhFixture, Large)   { runCase(20000, 3); }

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
