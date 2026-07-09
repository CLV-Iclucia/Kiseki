//
// test-sort.cc — RPK Sort end-to-end GPU tests
//

#include <RPK/sort.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace sim::rhi;
using namespace sim::rpk;

namespace {

struct SortFixture : public ::testing::Test {
    std::unique_ptr<Device> device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<Sort> sorter;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
        if (!device) GTEST_SKIP() << "No Vulkan device";

        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

        sorter = std::make_unique<Sort>(*device, *compiler);
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

TEST_F(SortFixture, KeysOnlySmall) {
    std::vector<uint32_t> data = {5, 3, 8, 1, 9, 2, 7, 4, 6, 0};
    const uint32_t N = static_cast<uint32_t>(data.size());

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->keys(*cmd, keysBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(keysBuf, N);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(result, data);
}

TEST_F(SortFixture, KeysOnlyRandom256) {
    constexpr uint32_t N = 256;
    std::vector<uint32_t> data(N);
    std::mt19937 rng(42);
    std::generate(data.begin(), data.end(), [&]() { return rng() % 10000; });

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->keys(*cmd, keysBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(keysBuf, N);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(result, data);
}

TEST_F(SortFixture, PairsSmall) {
    std::vector<uint32_t> keys   = {30, 10, 20, 50, 40};
    std::vector<uint32_t> values = { 0,  1,  2,  3,  4};  // original indices
    const uint32_t N = static_cast<uint32_t>(keys.size());

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    auto valsBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, keys.data(), N * sizeof(uint32_t));
    upload(valsBuf, values.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->pairs(*cmd, keysBuf, valsBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto sortedKeys = readback(keysBuf, N);
    auto sortedVals = readback(valsBuf, N);

    std::vector<uint32_t> expectedKeys = {10, 20, 30, 40, 50};
    std::vector<uint32_t> expectedVals = { 1,  2,  0,  4,  3};
    EXPECT_EQ(sortedKeys, expectedKeys);
    EXPECT_EQ(sortedVals, expectedVals);
}

TEST_F(SortFixture, PairsRandom1024) {
    constexpr uint32_t N = 1024;
    std::vector<uint32_t> keys(N);
    std::vector<uint32_t> values(N);
    std::mt19937 rng(123);
    std::generate(keys.begin(), keys.end(), [&]() { return rng(); });
    std::iota(values.begin(), values.end(), 0u);

    // Reference sort
    std::vector<uint32_t> refKeys = keys;
    std::vector<uint32_t> refVals = values;
    std::vector<uint32_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    std::sort(indices.begin(), indices.end(),
              [&](uint32_t a, uint32_t b) { return refKeys[a] < refKeys[b]; });
    std::vector<uint32_t> expectedKeys(N), expectedVals(N);
    for (uint32_t i = 0; i < N; ++i) {
        expectedKeys[i] = refKeys[indices[i]];
        expectedVals[i] = refVals[indices[i]];
    }

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    auto valsBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, keys.data(), N * sizeof(uint32_t));
    upload(valsBuf, values.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->pairs(*cmd, keysBuf, valsBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto sortedKeys = readback(keysBuf, N);
    auto sortedVals = readback(valsBuf, N);

    EXPECT_EQ(sortedKeys, expectedKeys);
    EXPECT_EQ(sortedVals, expectedVals);
}

TEST_F(SortFixture, KeysLargeRandom) {
    // Stress test: 10000 random keys
    constexpr uint32_t N = 10000;
    std::vector<uint32_t> data(N);
    std::mt19937 rng(7777);
    std::generate(data.begin(), data.end(), [&]() { return rng(); });

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->keys(*cmd, keysBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(keysBuf, N);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(result, data);
}

TEST_F(SortFixture, AlreadySorted) {
    constexpr uint32_t N = 64;
    std::vector<uint32_t> data(N);
    std::iota(data.begin(), data.end(), 0u);

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->keys(*cmd, keysBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(keysBuf, N);
    EXPECT_EQ(result, data);
}

TEST_F(SortFixture, ReverseSorted) {
    constexpr uint32_t N = 128;
    std::vector<uint32_t> data(N);
    for (uint32_t i = 0; i < N; ++i) data[i] = N - 1 - i;

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(keysBuf, data.data(), N * sizeof(uint32_t));

    auto cmd = device->beginCommands(QueueType::Compute);
    sorter->keys(*cmd, keysBuf, N);
    device->submitAndWait(*cmd, QueueType::Compute);

    auto result = readback(keysBuf, N);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(result, data);
}
