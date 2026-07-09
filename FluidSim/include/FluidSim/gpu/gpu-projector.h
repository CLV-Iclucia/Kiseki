// ============================================================================
// include/FluidSim/gpu/gpu-projector.h
//
// GPUProjector owns:
//   - Face-weight computation  (build-weights.hlsl)
//   - Poisson system assembly  (build-system.hlsl)
//   - Pressure→velocity projection (project.hlsl)
//   - A GpuPressureSolver (owns the actual linear solve)
//
// The linear solver is selected at construction based on SolverConfig and
// can be hot-swapped at runtime via updateConfig().
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-pressure-solver.h>
#include <FluidSim/gpu/gpu-shaders.h>
#include <RHI/rhi.h>
#include <filesystem>
#include <memory>

namespace fluid::gpu {

// ============================================================================
class GPUProjector {
public:
    GPUProjector(sim::rhi::Device& device,
                 const GPUGridState& grid,
                 const SolverConfig& config);

    // Build system + run solver + project velocities
    void solve(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);

    // Hot-swap solver if config changes (e.g. switch from Jacobi to PCG)
    void updateConfig(sim::rhi::Device& device,
                      const SolverConfig& config);

private:
    SolverConfig config_;

    // ===== Owned buffers (build/project phase) =====
    sim::rhi::BufferRef Adiag_;
    sim::rhi::BufferRef Aneighbour_[6];
    sim::rhi::BufferRef rhs_;
    sim::rhi::BufferRef active_;
    sim::rhi::BufferRef pressure_;
    sim::rhi::BufferRef faceWeightsU_, faceWeightsV_, faceWeightsW_;

    // ===== Delegated solver =====
    std::unique_ptr<GPUPressureSolver> solver_;

    BuildWeightsCS buildWeights_;
    BuildSystemCS buildSystem_;
    ProjectCS project_;

    // ===== Internal methods =====
    void buildWeightsAndSystem(sim::rhi::CommandList& cmd,
                               const GPUGridState& grid, Real dt);
    void projectVelocities(sim::rhi::CommandList& cmd,
                           GPUGridState& grid, Real dt);

    // Helper: assemble PressureSystem view over owned buffers
    PressureSystem makePressureSystem(Vec3i gridSize) const;
};

} // namespace fluid::gpu
