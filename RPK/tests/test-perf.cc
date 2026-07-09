//
// test-perf.cc — RPK performance benchmarks: GPU vs CPU timing comparison
//
// Not a correctness test — reports wall-clock times for visual comparison.
// Run with: --gtest_filter=Perf*
//

#include <RPK/rpk.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace sim::rhi;
using namespace sim::rpk;
using Clock = std::chrono::high_resolution_clock;

namespace {

struct PerfFixture : public ::testing::Test {
    std::unique_ptr<Device> device;
    std::unique_ptr<ShaderCompiler> compiler;
    std::unique_ptr<Reduce> reducer;
    std::unique_ptr<Scan> scanner;
    std::unique_ptr<Sort> sorter;

    void SetUp() override {
        device = Device::create({.backend = Backend::Vulkan, .enableValidation = false});
        if (!device) GTEST_SKIP() << "No Vulkan device";

        compiler = device->createShaderCompiler();
        if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

        reducer = std::make_unique<Reduce>(*device, *compiler);
        scanner = std::make_unique<Scan>(*device, *compiler);
        sorter  = std::make_unique<Sort>(*device, *compiler);
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

    // Re-upload data for repeated runs (sort is destructive)
    void reupload(const BufferRef& dst, const void* data, uint32_t sizeBytes) {
        upload(dst, data, sizeBytes);
    }
};

void printResult(const char* label, double gpuMs, double cpuMs) {
    double speedup = cpuMs / gpuMs;
    std::cout << "  [" << label << "] GPU: " << gpuMs << " ms | CPU: " << cpuMs
              << " ms | Speedup: " << speedup << "x" << std::endl;
}

}  // namespace

// ============================================================================
// Reduce benchmarks
// ============================================================================

TEST_F(PerfFixture, ReduceSum1M) {
    constexpr uint32_t N = 1'000'000;
    std::vector<float> data(N, 1.0f);

    auto input  = makeDeviceBuffer(N * sizeof(float));
    auto output = makeDeviceBuffer(sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    // Warmup
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float32, input, output, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    // GPU timing (batched: multiple dispatches in one cmd)
    constexpr int kRuns = 50;
    auto t0 = Clock::now();
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        for (int i = 0; i < kRuns; ++i) {
            reducer->run(*cmd, ReduceOp::Sum, ScalarType::Float32, input, output, N);
        }
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    // CPU timing
    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        volatile float sum = 0.0f;
        for (uint32_t j = 0; j < N; ++j) sum += data[j];
        (void)sum;
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Reduce Sum 1M floats (batched)", gpuMs, cpuMs);
}

TEST_F(PerfFixture, ReduceMax4M) {
    constexpr uint32_t N = 4'000'000;
    std::vector<float> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });

    auto input  = makeDeviceBuffer(N * sizeof(float));
    auto output = makeDeviceBuffer(sizeof(float));
    upload(input, data.data(), N * sizeof(float));

    // Warmup
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        reducer->run(*cmd, ReduceOp::Max, ScalarType::Float32, input, output, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    constexpr int kRuns = 20;
    auto t0 = Clock::now();
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        for (int i = 0; i < kRuns; ++i) {
            reducer->run(*cmd, ReduceOp::Max, ScalarType::Float32, input, output, N);
        }
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        volatile float mx = -1e30f;
        for (uint32_t j = 0; j < N; ++j) {
            if (data[j] > mx) mx = data[j];
        }
        (void)mx;
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Reduce Max 4M floats (batched)", gpuMs, cpuMs);
}

// ============================================================================
// Scan benchmark — only meaningful at large scale where GPU bandwidth wins
// ============================================================================

TEST_F(PerfFixture, ExclusiveScan16M) {
    constexpr uint32_t N = 16'000'000;
    std::vector<uint32_t> data(N, 1u);

    auto input  = makeDeviceBuffer(N * sizeof(uint32_t));
    auto output = makeDeviceBuffer(N * sizeof(uint32_t));
    upload(input, data.data(), N * sizeof(uint32_t));

    // Warmup
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    // GPU: batched
    constexpr int kRuns = 10;
    auto t0 = Clock::now();
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        for (int i = 0; i < kRuns; ++i) {
            scanner->exclusive(*cmd, ScanOp::Sum, ScalarType::Uint32, input, output, N);
        }
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    // CPU
    std::vector<uint32_t> cpuOut(N);
    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        std::exclusive_scan(data.begin(), data.end(), cpuOut.begin(), 0u);
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Exclusive Scan 16M uint32 (batched)", gpuMs, cpuMs);
}

// ============================================================================
// Sort benchmarks
// ============================================================================

TEST_F(PerfFixture, SortKeys256K) {
    constexpr uint32_t N = 256'000;
    std::vector<uint32_t> data(N);
    std::mt19937 rng(1234);
    std::generate(data.begin(), data.end(), [&]() { return rng(); });

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));

    // Warmup
    upload(keysBuf, data.data(), N * sizeof(uint32_t));
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->keys(*cmd, keysBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    // GPU timing (re-upload each run since sort is destructive)
    constexpr int kRuns = 5;
    auto t0 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        reupload(keysBuf, data.data(), N * sizeof(uint32_t));
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->keys(*cmd, keysBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    // CPU timing
    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        std::vector<uint32_t> tmp = data;
        std::sort(tmp.begin(), tmp.end());
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Sort Keys 256K uint32", gpuMs, cpuMs);
}

TEST_F(PerfFixture, SortKeys1M) {
    constexpr uint32_t N = 1'000'000;
    std::vector<uint32_t> data(N);
    std::mt19937 rng(5678);
    std::generate(data.begin(), data.end(), [&]() { return rng(); });

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));

    // Warmup
    upload(keysBuf, data.data(), N * sizeof(uint32_t));
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->keys(*cmd, keysBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    constexpr int kRuns = 5;
    auto t0 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        reupload(keysBuf, data.data(), N * sizeof(uint32_t));
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->keys(*cmd, keysBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        std::vector<uint32_t> tmp = data;
        std::sort(tmp.begin(), tmp.end());
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Sort Keys 1M uint32", gpuMs, cpuMs);
}

TEST_F(PerfFixture, SortPairs1M) {
    constexpr uint32_t N = 1'000'000;
    std::vector<uint32_t> keys(N), values(N);
    std::mt19937 rng(9999);
    std::generate(keys.begin(), keys.end(), [&]() { return rng(); });
    std::iota(values.begin(), values.end(), 0u);

    auto keysBuf = makeDeviceBuffer(N * sizeof(uint32_t));
    auto valsBuf = makeDeviceBuffer(N * sizeof(uint32_t));

    // Warmup
    upload(keysBuf, keys.data(), N * sizeof(uint32_t));
    upload(valsBuf, values.data(), N * sizeof(uint32_t));
    {
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->pairs(*cmd, keysBuf, valsBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }

    constexpr int kRuns = 5;
    auto t0 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        reupload(keysBuf, keys.data(), N * sizeof(uint32_t));
        reupload(valsBuf, values.data(), N * sizeof(uint32_t));
        auto cmd = device->beginCommands(QueueType::Compute);
        sorter->pairs(*cmd, keysBuf, valsBuf, N);
        device->submitAndWait(*cmd, QueueType::Compute);
    }
    auto t1 = Clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRuns;

    // CPU: sort pairs by key using index array
    auto t2 = Clock::now();
    for (int i = 0; i < kRuns; ++i) {
        std::vector<uint32_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(),
                  [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    }
    auto t3 = Clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count() / kRuns;

    printResult("Sort Pairs 1M uint32", gpuMs, cpuMs);
}
