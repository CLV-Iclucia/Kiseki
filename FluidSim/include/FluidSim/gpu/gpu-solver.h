// ============================================================================
// include/FluidSim/gpu/gpu-solver.h
// GPUModularSolver: GPU 后端的中间层基类
// ============================================================================
#pragma once

#include <FluidSim/fluid-solver.h>
#include <FluidSim/gpu/gpu-context.h>

namespace fluid::gpu {

class GPUModularSolver : public FluidModularSolver {
public:
    using FluidModularSolver::FluidModularSolver;

    void solve(FluidContext& ctx, Real dt) final {
        solve(static_cast<GPUFluidContext&>(ctx), dt);
    }

    virtual void solve(GPUFluidContext& ctx, Real dt) = 0;
};

} // namespace fluid::gpu
