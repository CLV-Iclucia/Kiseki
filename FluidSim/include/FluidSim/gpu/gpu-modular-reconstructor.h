// ============================================================================
// include/FluidSim/gpu/gpu-modular-reconstructor.h
// GPU Reconstructor adapted as GPUModularSolver subclass
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-solver.h>
#include <FluidSim/gpu/gpu-reconstructor.h>
#include <memory>

namespace fluid::gpu {

class GPUReconstructorSolver : public GPUModularSolver {
public:
    GPUReconstructorSolver(GPUFluidContext& ctx,
                           std::shared_ptr<GPUReconstructor> reconImpl)
        : GPUModularSolver("gpu_reconstructor"), impl_(std::move(reconImpl))
    {
        (void)ctx;
    }

    void solve(GPUFluidContext& ctx, Real dt) override;

private:
    std::shared_ptr<GPUReconstructor> impl_;
};

} // namespace fluid::gpu
