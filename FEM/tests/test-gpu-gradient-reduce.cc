// ============================================================================
// test-gpu-gradient-reduce.cc — validate GpuGradientReduce: deterministic,
// atomic-free scatter-add of scattered (row, vec3) entries into a dense
// per-vertex gradient.
//
// We pre-fill g with a random base, push many random (row, vec3) entries
// (multiple per row, some vertices left untouched), reduce on the GPU, and
// compare against the CPU reference base[v] + sum of all entries with row==v.
// Untouched vertices must keep their base value unchanged.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-gradient-reduce.h>
#include <Maths/block-vector.h>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace sim;
using namespace sim::rhi;
using namespace sim::fem::gpu;

namespace {

struct GradReduceFixture : public ::testing::Test {
    std::unique_ptr<Device>            device;
    std::unique_ptr<ShaderCompiler>    compiler;
    std::unique_ptr<GpuGradientReduce> reduce;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";
        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";
        reduce = std::make_unique<GpuGradientReduce>(*device, *compiler);
        if (!reduce->valid()) GTEST_SKIP() << "GPU pipeline failed to compile";
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

TEST_F(GradReduceFixture, ScatterAddMatchesCpu) {
    const uint32_t V = 200;     // vertices
    const uint32_t E = 1000;    // scattered entries

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<uint32_t> vd(0, V - 1);

    // Base gradient (e.g. the elastic gradient already in g).
    std::vector<double> base(size_t(V) * 3);
    for (auto& v : base) v = u(rng);

    // Scattered (row, vec3) entries; bias rows to a sub-range so many vertices
    // are touched multiple times and some (>= V/2) are never touched.
    std::vector<uint32_t> row(E);
    std::vector<double>   val(size_t(E) * 3);
    std::uniform_int_distribution<uint32_t> vdLow(0, V / 2 - 1);
    for (uint32_t e = 0; e < E; ++e) {
        row[e] = vdLow(rng);
        val[e * 3 + 0] = u(rng);
        val[e * 3 + 1] = u(rng);
        val[e * 3 + 2] = u(rng);
    }

    // ---- CPU reference ----
    std::vector<double> ref = base;
    for (uint32_t e = 0; e < E; ++e)
        for (int d = 0; d < 3; ++d)
            ref[size_t(row[e]) * 3 + d] += val[e * 3 + d];

    // ---- GPU ----
    auto bG   = makeBuf(size_t(V) * 3 * sizeof(double));
    auto bRow = makeBuf(size_t(E) * sizeof(uint32_t));
    auto bVal = makeBuf(size_t(E) * 3 * sizeof(double));
    upload(bG, base.data(), base.size() * sizeof(double));
    upload(bRow, row.data(), row.size() * sizeof(uint32_t));
    upload(bVal, val.data(), val.size() * sizeof(double));

    reduce->addInto(bG, bRow, bVal, E);

    auto gpu = download<double>(bG, size_t(V) * 3);

    double maxErr = 0.0, refMax = 0.0;
    uint32_t touched = 0, untouchedExact = 0;
    std::vector<char> isTouched(V, 0);
    for (uint32_t e = 0; e < E; ++e) isTouched[row[e]] = 1;
    for (uint32_t i = 0; i < V; ++i) {
        for (int d = 0; d < 3; ++d) {
            double a = gpu[i * 3 + d], b = ref[i * 3 + d];
            maxErr = std::max(maxErr, std::abs(a - b));
            refMax = std::max(refMax, std::abs(b));
        }
        if (isTouched[i]) ++touched;
        else {
            bool same = gpu[i*3+0] == base[i*3+0] && gpu[i*3+1] == base[i*3+1] &&
                        gpu[i*3+2] == base[i*3+2];
            if (same) ++untouchedExact;
        }
    }
    double relErr = maxErr / std::max(refMax, 1e-30);
    spdlog::info("[test-gpu-gradient-reduce] V={} E={} touched={} relErr={:.3e}",
                 V, E, touched, relErr);

    EXPECT_LT(relErr, 1e-12);                 // pure double adds -> near-exact
    EXPECT_EQ(untouchedExact, V - touched);   // untouched rows keep base exactly
    EXPECT_GT(touched, 0u);
    EXPECT_LT(touched, V);                    // confirm some rows untouched
}

#endif // FEM_GPU_ENABLED
// When RHI is disabled this TU is empty; GTest::gtest_main provides main().
