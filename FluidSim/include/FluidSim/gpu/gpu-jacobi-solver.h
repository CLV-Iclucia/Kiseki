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
#include <FluidSim/gpu/gpu-shaders.h>

namespace fluid::gpu
{
    class GPUJacobiSolver final : public GPUPressureSolver
    {
    public:
        GPUJacobiSolver(sim::rhi::Device& device,
                        const PressureSystem& system,
                        const SolverConfig& config);

        void solve(sim::rhi::CommandList& cmd,
                   const PressureSystem& system) override;

        void updateConfig(const SolverConfig& config) override
        {
            iterations_ = config.pressureMaxIters;
        }

    private:
        int iterations_;

        // Ping-pong buffer (owned by solver)
        sim::rhi::BufferRef pressurePing_; // extra buffer; system.pressure is pong

        JacobiIterCS jacobi_;
    };
} // namespace fluid::gpu
