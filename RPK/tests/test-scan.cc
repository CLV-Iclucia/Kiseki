//
// test-scan.cc — RPK Scan end-to-end GPU tests
//

#include <RPK/scan.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

using namespace sim::rhi;
using namespace sim::rpk;

namespace {

struct ScanFixture : public ::testing::Test {
    std::unique_ptr<Device> device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<Scan> scanner;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";

        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

        scanner = std::make_unique<Scan>(*device, *compiler);
    }

    BufferRef makeDeviceBuffer(uint32_t sizeBytes) {
        return device->createBuffer({
            .sizeBytes  = sizeBytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        });
    }

    void upload(const BufferRef& dst, const void* data, uint32_t sizeBytes) {
        auto staging = device->createBuffer({
            .sizeBytes  = sizeBytes,
            .visibility = BufferDesc::Visibility::HostVisible,
            .usage      = BufferDesc::TransferSrc,
        });
        void* ptr = staging->map();
        std::memcpy(ptr, data, sizeBytes);
        staging->unmap();

        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, sizeBytes}}};
        cmd->copyBuffer(staging, dst, region);
        device->submitAndWait(*cmd, QueueType::Transfer);
    }

    std::vector<uint32_t> readback(const BufferRef& buf, uint32_t count) {
        uint32_t bytes = count * sizeof(uint32_t);
        auto rb = device->createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::Readback,
            .usage      = BufferDesc::TransferDst,
        });
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
        cmd->copyBuffer(buf, rb, region);
        device->submitAndWait(*cmd, QueueType::Transfer);

        auto span = rb->mapTyped<uint32_t>();
        std::vector<uint32_t> result(span.begin(), span.begin() + count);
        rb->unmap();
        return result;
    }
};

}  // namespace

TEST_F(ScanFixture, ExclusiveSumSmall) {
    // [1,1,1,1,1,1,1,1] → exclusive sum → [0,1,2,3,4,5,6,7]
    constexpr uint32_t N = 8;
    std::vector<uint32_t> data(N, 1u);

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(output, N);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(result[i], i) << "mismatch at i=" << i;
    }
}

TEST_F(ScanFixture, ExclusiveSumMedium) {
    // [3,3,3,...,3] (512 elements) → exclusive sum → [0,3,6,9,...,1533]
    constexpr uint32_t N = 512;
    std::vector<uint32_t> data(N, 3u);

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(output, N);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(result[i], i * 3u) << "mismatch at i=" << i;
    }
}

TEST_F(ScanFixture, ExclusiveSumLarge) {
    // Exercises multi-level recursion: 2048 elements
    constexpr uint32_t N = 2048;
    std::vector<uint32_t> data(N, 1u);

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(output, N);
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(result[i], i) << "mismatch at i=" << i;
    }
}

TEST_F(ScanFixture, InclusiveSumSmall) {
    // [1,2,3,4] → inclusive sum → [1,3,6,10]
    constexpr uint32_t N = 4;
    std::vector<uint32_t> data = {1, 2, 3, 4};

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    scanner->inclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(output, N);
    std::vector<uint32_t> expected = {1, 3, 6, 10};
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(result[i], expected[i]) << "mismatch at i=" << i;
    }
}

TEST_F(ScanFixture, ExclusiveSumVeryLarge) {
    // 100000 elements → verifies deep recursion
    constexpr uint32_t N = 100000;
    std::vector<uint32_t> data(N, 2u);

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(output, N);
    // Spot check some values
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 2u);
    EXPECT_EQ(result[100], 200u);
    EXPECT_EQ(result[N - 1], (N - 1) * 2u);
}
