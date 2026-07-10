//
// test-reduce.cc — RPK Reduce end-to-end GPU tests
//

#include <RPK/reduce.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace ksk::rhi;
using namespace ksk::rpk;

namespace {

struct ReduceFixture : public ::testing::Test {
    std::unique_ptr<Device> device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<Reduce> reducer;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";

        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

        reducer = std::make_unique<Reduce>(*device, *compiler);
    }

    BufferRef makeDeviceBuffer(uint32_t count, uint32_t elemSize) {
        return device->createBuffer({
            .sizeBytes  = count * elemSize,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        });
    }

    BufferRef makeReadback(uint32_t sizeBytes) {
        return device->createBuffer({
            .sizeBytes  = sizeBytes,
            .visibility = BufferDesc::Visibility::Readback,
            .usage      = BufferDesc::TransferDst,
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

    float readbackFloat(const BufferRef& buf) {
        auto rb = makeReadback(sizeof(float));
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, sizeof(float)}}};
        cmd->copyBuffer(buf, rb, region);
        device->submitAndWait(*cmd, QueueType::Transfer);
        auto span = rb->mapTyped<float>();
        float val = span[0];
        rb->unmap();
        return val;
    }

    uint32_t readbackUint(const BufferRef& buf) {
        auto rb = makeReadback(sizeof(uint32_t));
        auto cmd = device->beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, sizeof(uint32_t)}}};
        cmd->copyBuffer(buf, rb, region);
        device->submitAndWait(*cmd, QueueType::Transfer);
        auto span = rb->mapTyped<uint32_t>();
        uint32_t val = span[0];
        rb->unmap();
        return val;
    }
};

}  // namespace

TEST_F(ReduceFixture, SumSmall) {
    // 16 elements: 1,2,3,...,16 → sum = 136
    constexpr uint32_t N = 16;
    std::vector<float> data(N);
    std::iota(data.begin(), data.end(), 1.0f);

    auto input  = makeDeviceBuffer(N, sizeof(float));
    auto output = makeDeviceBuffer(1, sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    float result = readbackFloat(output);
    float expected = N * (N + 1.0f) / 2.0f;
    EXPECT_NEAR(result, expected, 1e-3f);
}

TEST_F(ReduceFixture, SumLarge) {
    // 10000 elements of 1.0f → sum = 10000.0f
    constexpr uint32_t N = 10000;
    std::vector<float> data(N, 1.0f);

    auto input  = makeDeviceBuffer(N, sizeof(float));
    auto output = makeDeviceBuffer(1, sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    float result = readbackFloat(output);
    EXPECT_NEAR(result, 10000.0f, 1.0f);  // float accumulation tolerance
}

TEST_F(ReduceFixture, MaxFloat) {
    constexpr uint32_t N = 1024;
    std::vector<float> data(N);
    for (uint32_t i = 0; i < N; ++i) data[i] = static_cast<float>(i) - 500.0f;
    // Max = 523.0f

    auto input  = makeDeviceBuffer(N, sizeof(float));
    auto output = makeDeviceBuffer(1, sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Max, ScalarType::Float32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    float result = readbackFloat(output);
    EXPECT_FLOAT_EQ(result, 523.0f);
}

TEST_F(ReduceFixture, MinFloat) {
    constexpr uint32_t N = 512;
    std::vector<float> data(N);
    for (uint32_t i = 0; i < N; ++i) data[i] = static_cast<float>(i) + 10.0f;
    data[300] = -999.0f;  // plant the minimum

    auto input  = makeDeviceBuffer(N, sizeof(float));
    auto output = makeDeviceBuffer(1, sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Min, ScalarType::Float32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    float result = readbackFloat(output);
    EXPECT_FLOAT_EQ(result, -999.0f);
}

TEST_F(ReduceFixture, SumUint32) {
    constexpr uint32_t N = 256;
    std::vector<uint32_t> data(N, 3u);  // sum = 768

    auto input  = makeDeviceBuffer(N, sizeof(uint32_t));
    auto output = makeDeviceBuffer(1, sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Uint32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    uint32_t result = readbackUint(output);
    EXPECT_EQ(result, 768u);
}

TEST_F(ReduceFixture, MultiLevel) {
    // Force multi-level: > 256*256 = 65536 elements
    constexpr uint32_t N = 70000;
    std::vector<float> data(N, 1.0f);

    auto input  = makeDeviceBuffer(N, sizeof(float));
    auto output = makeDeviceBuffer(1, sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float32, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    float result = readbackFloat(output);
    EXPECT_NEAR(result, 70000.0f, 10.0f);
}

TEST_F(ReduceFixture, SumFloat64) {
    // 1000 elements of 0.001 → sum = 1.0 (tests double precision)
    constexpr uint32_t N = 1000;
    std::vector<double> data(N, 0.001);

    auto input  = makeDeviceBuffer(N, sizeof(double));
    auto output = makeDeviceBuffer(1, sizeof(double));
    upload(input, data.data(), N * sizeof(double));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float64, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    // Readback double
    auto rb = makeReadback(sizeof(double));
    auto cmd2 = device->beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
    cmd2->copyBuffer(output, rb, region);
    device->submitAndWait(*cmd2, QueueType::Transfer);
    auto span = rb->mapTyped<double>();
    double result = span[0];
    rb->unmap();

    EXPECT_NEAR(result, 1.0, 1e-10);
}

TEST_F(ReduceFixture, MaxFloat64) {
    constexpr uint32_t N = 512;
    std::vector<double> data(N);
    for (uint32_t i = 0; i < N; ++i) data[i] = static_cast<double>(i) * 0.1;
    data[77] = 99999.5;  // plant the max

    auto input  = makeDeviceBuffer(N, sizeof(double));
    auto output = makeDeviceBuffer(1, sizeof(double));
    upload(input, data.data(), N * sizeof(double));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Max, ScalarType::Float64, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto rb = makeReadback(sizeof(double));
    auto cmd2 = device->beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
    cmd2->copyBuffer(output, rb, region);
    device->submitAndWait(*cmd2, QueueType::Transfer);
    auto span = rb->mapTyped<double>();
    double result = span[0];
    rb->unmap();

    EXPECT_DOUBLE_EQ(result, 99999.5);
}

TEST_F(ReduceFixture, SumFloat64Large) {
    // 100000 elements — tests multi-level reduce with double
    constexpr uint32_t N = 100000;
    std::vector<double> data(N, 1.0);

    auto input  = makeDeviceBuffer(N, sizeof(double));
    auto output = makeDeviceBuffer(1, sizeof(double));
    upload(input, data.data(), N * sizeof(double));

    auto cmd = device->beginCommands(QueueType::Compute);
    reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float64, input, output, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto rb = makeReadback(sizeof(double));
    auto cmd2 = device->beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
    cmd2->copyBuffer(output, rb, region);
    device->submitAndWait(*cmd2, QueueType::Transfer);
    auto span = rb->mapTyped<double>();
    double result = span[0];
    rb->unmap();

    EXPECT_DOUBLE_EQ(result, 100000.0);
}
