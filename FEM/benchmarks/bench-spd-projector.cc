// ============================================================================
// FEM/benchmarks/bench-spd-projector.cc
// Throughput benchmark for the GPU Jacobi 9x9 SPD eigen-filter.
// Compares cyclic vs classical strategies and iteration-count settings, to
// support ongoing kernel optimization.
//
// Note: timings are end-to-end project() calls (upload + dispatch + readback).
// Transfer overhead is identical across variants, so relative comparisons of
// strategy / iteration count remain meaningful. For a large batch the GPU
// compute dominates.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/gpu/gpu-spd-projector.h>
#include <RHI/rhi.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace ksk::fem::gpu;
using Clock = std::chrono::high_resolution_clock;

namespace {

std::vector<double> makeRandomSymmetric(int count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> mats(size_t(count) * 81);
    for (int m = 0; m < count; ++m) {
        double A[9][9];
        for (int i = 0; i < 9; ++i)
            for (int j = i; j < 9; ++j) { double v = u(rng); A[i][j] = v; A[j][i] = v; }
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j) mats[size_t(m) * 81 + i * 9 + j] = A[i][j];
    }
    return mats;
}

double timeVariant(GpuSpdProjector9& proj, GpuSpdProjector9::Strategy strat,
                   GpuSpdProjector9::Clamp clamp, const std::vector<double>& in,
                   int count, int repeats) {
    std::vector<double> out;
    proj.project(strat, clamp, in, count, out);  // warmup
    auto t0 = Clock::now();
    for (int r = 0; r < repeats; ++r)
        proj.project(strat, clamp, in, count, out);
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

void runConfig(ksk::rhi::Device& device, ksk::rhi::ShaderCompiler& compiler,
               const std::vector<double>& in, int count, int repeats,
               int cyclicSweeps, int classicalRots) {
    GpuSpdProjector9 proj(device, compiler, {}, cyclicSweeps, classicalRots);
    if (!proj.valid()) { std::cerr << "  projector invalid\n"; return; }

    struct V { const char* name; GpuSpdProjector9::Strategy s; GpuSpdProjector9::Clamp c; };
    V variants[] = {
        {"cyclic/abs",    GpuSpdProjector9::Strategy::Cyclic,    GpuSpdProjector9::Clamp::Abs},
        {"classical/abs", GpuSpdProjector9::Strategy::Classical, GpuSpdProjector9::Clamp::Abs},
    };

    std::cout << "  [sweeps=" << cyclicSweeps << " rotations=" << classicalRots << "]\n";
    for (auto& v : variants) {
        double ms = timeVariant(proj, v.s, v.c, in, count, repeats);
        double mps = (count / 1e6) / (ms / 1e3);  // million matrices / sec
        std::cout << "    " << std::left << std::setw(16) << v.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << ms << " ms   "
                  << std::setprecision(2) << std::setw(8) << mps << " M-mat/s\n";
    }
}

} // namespace

int main() {
    auto device = ksk::rhi::Device::create({.backend = ksk::rhi::Backend::Vulkan,
                                            .enableValidation = false});
    if (!device) { std::cerr << "No Vulkan device\n"; return 0; }
    auto compiler = device->createShaderCompiler();
    if (!compiler) { std::cerr << "dxcompiler unavailable\n"; return 0; }

    const int count   = 200000;
    const int repeats = 10;
    auto in = makeRandomSymmetric(count, 1234);

    std::cout << "GPU SPD projector benchmark — " << count << " symmetric 9x9 matrices, "
              << repeats << " repeats/avg\n";

    runConfig(*device, *compiler, in, count, repeats, /*sweeps*/ 6,  /*rots*/ 200);
    runConfig(*device, *compiler, in, count, repeats, /*sweeps*/ 10, /*rots*/ 400);
    runConfig(*device, *compiler, in, count, repeats, /*sweeps*/ 15, /*rots*/ 600);

    return 0;
}

#else
int main() { return 0; }  // RHI disabled
#endif // FEM_GPU_ENABLED
