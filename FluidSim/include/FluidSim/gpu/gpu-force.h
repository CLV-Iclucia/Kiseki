// ============================================================================
// include/FluidSim/gpu/gpu-force.h
// GPU Body Force Solver as GPUModularSolver subclass
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-solver.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

struct GPUForceConfig {
    Vec3d gravity{0.0, -9.8, 0.0};
};

class GPUForceSolver : public GPUModularSolver {
public:
    GPUForceSolver(GPUFluidContext& ctx, GPUForceConfig config = {});

    void solve(GPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    GPUForceConfig config_;
    sim::rhi::PipelineRef psoBodyForce_;
};

} // namespace fluid::gpu
