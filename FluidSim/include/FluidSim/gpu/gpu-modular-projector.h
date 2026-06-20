// ============================================================================
// include/FluidSim/gpu/gpu-modular-projector.h
// GPU Projector adapted as GPUModularSolver subclass
// Wraps existing GPUProjector internally
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-solver.h>
#include <FluidSim/gpu/gpu-projector.h>
#include <memory>

namespace fluid::gpu {

class GPUProjectorSolver : public GPUModularSolver {
public:
    GPUProjectorSolver(GPUFluidContext& ctx,
                       std::shared_ptr<GPUProjector> projImpl)
        : GPUModularSolver("gpu_projector"), impl_(std::move(projImpl))
    {
        (void)ctx;
    }

    void solve(GPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    std::shared_ptr<GPUProjector> impl_;
};

} // namespace fluid::gpu
