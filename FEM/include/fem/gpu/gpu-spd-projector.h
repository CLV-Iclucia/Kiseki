// ============================================================================
// FEM/include/fem/gpu/gpu-spd-projector.h
// Batch SPD projection (eigenvalue filtering) of symmetric 9x9 matrices on GPU
// via the Jacobi eigenvalue algorithm. Modular & independently testable.
//
// Two iteration strategies (cyclic / classical) and two clamp modes
// (abs / psd) are compiled as separate pipeline variants.
// ============================================================================
#pragma once

#include <RHI/rhi.h>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace sim::fem::gpu {

class GpuSpdProjector9 {
public:
    enum class Strategy { Cyclic = 0, Classical = 1 };
    enum class Clamp    { Abs = 0,    Psd = 1 };

    GpuSpdProjector9(sim::rhi::Device& device,
                     sim::rhi::ShaderCompiler& compiler,
                     const std::filesystem::path& shaderDir = {},
                     int cyclicSweeps = 10,
                     int classicalRotations = 400);

    [[nodiscard]] bool valid() const { return valid_; }

    // Filter `count` symmetric 9x9 matrices, row-major flattened (count*81).
    // `out` is resized to match. Uses the requested strategy/clamp variant.
    void project(Strategy strategy, Clamp clamp,
                 const std::vector<double>& matsIn, int count,
                 std::vector<double>& matsOut);

private:
    sim::rhi::Device& device_;
    bool valid_ = false;
    sim::rhi::PipelineRef pso_[2][2];  // [strategy][clamp]

};

} // namespace sim::fem::gpu
