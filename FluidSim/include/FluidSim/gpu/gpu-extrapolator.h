// ============================================================================
// include/FluidSim/gpu/gpu-extrapolator.h
// GPU Extrapolator + Force as GPUModularSolver subclass
// These use the boundary pipelines previously owned by GPUFluidBackend
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-shaders.h>
#include <FluidSim/gpu/gpu-solver.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUExtrapolatorSolver : public GPUModularSolver {
public:
    GPUExtrapolatorSolver(GPUFluidContext& ctx, int iters = 10);

    void solve(GPUFluidContext& ctx, Real dt) override;

private:
    int iters_;
    ExtrapolateCS extrapolate_;
};

} // namespace fluid::gpu
