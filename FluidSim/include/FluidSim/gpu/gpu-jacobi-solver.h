// ============================================================================
// include/FluidSim/gpu/gpu-jacobi-solver.h
// Jacobi iterative pressure solver (ping-pong buffer scheme).
//
// Each iteration:
//   p_new[i] = (rhs[i] - sum_j(A[i,j] * p_old[j])) / A[i,i]
//
// Convergence: slow (spectral radius ≈ 1 - O(h²)) but trivially parallel.
// Adequate for 64³ grids with ~100-200 iterations.
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-pressure-solver.h>

namespace fluid::gpu {

SHADER_PARAMS_BEGIN(JacobiIterParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, pressureIn);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, pressureOut);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour0);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour1);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour2);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour3);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour4);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour5);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, rhs);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
SHADER_PARAMS_END();

class GpuJacobiSolver final : public GpuPressureSolver {
public:
    GpuJacobiSolver(sim::rhi::Device&         device,
                    sim::rhi::ShaderCompiler& compiler,
                    const PressureSystem&     system,
                    const SolverConfig&       config,
                    const std::filesystem::path& shaderDir);

    void solve(sim::rhi::CommandList& cmd,
               const PressureSystem& system) override;

    void updateConfig(const SolverConfig& config) override {
        iterations_ = config.pressureMaxIters;
    }

private:
    int iterations_;

    // Ping-pong buffer (owned by solver)
    sim::rhi::BufferRef pressurePing_;  // extra buffer; system.pressure is pong

    sim::rhi::PipelineRef psoJacobi_;
};

} // namespace fluid::gpu
