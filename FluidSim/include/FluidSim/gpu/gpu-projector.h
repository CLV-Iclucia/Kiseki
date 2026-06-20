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
#include <RHI/rhi.h>
#include <filesystem>
#include <memory>

namespace fluid::gpu {

// SHADER_PARAMS for build / project steps (solver-independent)

SHADER_PARAMS_BEGIN(BuildWeightsParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    faceWeightsU);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    faceWeightsV);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    faceWeightsW);
    SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, colliderSdf);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAM_SCALAR(float,    gridSpacing);
    SHADER_PARAM_SCALAR(float,    originX);
    SHADER_PARAM_SCALAR(float,    originY);
    SHADER_PARAM_SCALAR(float,    originZ);
    SHADER_PARAM_SCALAR(uint32_t, maxFaces);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(BuildSystemParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Adiag);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour0);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour1);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour2);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour3);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour4);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    Aneighbour5);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    rhs);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef,    active);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    uGrid);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    vGrid);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    wGrid);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    faceWeightsU);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    faceWeightsV);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef,    faceWeightsW);
    SHADER_PARAM_IMAGE (sim::rhi::ImageBinding, fluidSdf);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAM_SCALAR(float,    gridSpacing);
    SHADER_PARAM_SCALAR(float,    density);
    SHADER_PARAM_SCALAR(float,    dt);
    SHADER_PARAM_SCALAR(float,    originX);
    SHADER_PARAM_SCALAR(float,    originY);
    SHADER_PARAM_SCALAR(float,    originZ);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ProjectParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, uGrid);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, vGrid);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, wGrid);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, pressure);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsU);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsV);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, faceWeightsW);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAM_SCALAR(float,    gridSpacing);
    SHADER_PARAM_SCALAR(float,    density);
    SHADER_PARAM_SCALAR(float,    dt);
    SHADER_PARAM_SCALAR(uint32_t, maxFaces);
SHADER_PARAMS_END();

// ============================================================================
class GPUProjector {
public:
    GPUProjector(sim::rhi::Device&             device,
                 sim::rhi::ShaderCompiler&     compiler,
                 const GPUGridState&           grid,
                 const SolverConfig&           config,
                 const std::filesystem::path&  shaderDir);

    // Build system + run solver + project velocities
    void solve(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);

    // Hot-swap solver if config changes (e.g. switch from Jacobi to PCG)
    void updateConfig(sim::rhi::Device&         device,
                      sim::rhi::ShaderCompiler& compiler,
                      const SolverConfig&       config,
                      const std::filesystem::path& shaderDir);

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
    std::unique_ptr<GpuPressureSolver> solver_;

    // ===== Pipelines (build / project only) =====
    sim::rhi::PipelineRef psoBuildWeights_;
    sim::rhi::PipelineRef psoBuildSystem_;
    sim::rhi::PipelineRef psoProject_;

    // ===== Internal methods =====
    void buildWeightsAndSystem(sim::rhi::CommandList& cmd,
                               const GPUGridState& grid, Real dt);
    void projectVelocities(sim::rhi::CommandList& cmd,
                           GPUGridState& grid, Real dt);

    // Helper: assemble PressureSystem view over owned buffers
    PressureSystem makePressureSystem(Vec3i gridSize) const;
};

} // namespace fluid::gpu
