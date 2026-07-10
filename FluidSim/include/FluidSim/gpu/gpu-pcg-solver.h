// ============================================================================
// include/FluidSim/gpu/gpu-pcg-solver.h
// Jacobi-preconditioned CG pressure solver (fully GPU-resident, float32).
// All scalar operations (alpha, beta) are computed on-GPU without readback.
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-pressure-solver.h>
#include <FluidSim/gpu/gpu-shaders.h>

namespace fluid::gpu {

// ============================================================================
class GpuPCGSolver final : public GPUPressureSolver {
public:
    GpuPCGSolver(ksk::rhi::Device& device,
                 const PressureSystem& system,
                 const SolverConfig& config);

    void solve(ksk::rhi::CommandList& cmd,
               const PressureSystem& system) override;

    void updateConfig(const SolverConfig& config) override {
        maxIters_ = config.pressureMaxIters;
    }

private:
    int maxIters_;

    // CG working vectors
    ksk::rhi::BufferRef cgR_, cgZ_, cgS_;

    // Scalar buffers for alpha/beta (GPU-side, no CPU readback)
    ksk::rhi::BufferRef reduceBuf_;       // partial sums (one per workgroup)
    ksk::rhi::BufferRef sigmaScalar_;     // σ = dot(z, r)
    ksk::rhi::BufferRef dotSZScalar_;     // dot(s, z) → denominator for α
    ksk::rhi::BufferRef alphaScalar_;     // α = σ / dot(s,z)
    ksk::rhi::BufferRef sigmaNewScalar_;  // σ_new for β

    PCGSpMVCS spmv_;
    PCGDotCS dotProduct_;
    PCGReduceFinalCS reduceFinal_;
    PCGSaxpyCS saxpy_;
    PCGJacobiPrecondCS jacobiPrecond_;
    PCGScalarDivCS scalarDiv_;
    PCGUpdateSCS updateS_;
};

} // namespace fluid::gpu
