// ============================================================================
// include/FluidSim/gpu/gpu-modular-advector.h
// GPU Advector adapted as GPUModularSolver subclasses
// Wraps existing GPUAdvector internally
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-solver.h>
#include <FluidSim/gpu/gpu-advector.h>
#include <memory>

namespace fluid::gpu {

// ---- P2G Solver (scatter particles to grid) ----
class GPUAdvectorP2GSolver : public GPUModularSolver {
public:
    GPUAdvectorP2GSolver(GPUFluidContext& ctx,
                         std::shared_ptr<GPUAdvector> advImpl)
        : GPUModularSolver("gpu_advector_p2g"), impl_(std::move(advImpl))
    {
        (void)ctx;
    }

    void solve(GPUFluidContext& ctx, Real dt) override;

private:
    std::shared_ptr<GPUAdvector> impl_;
};

// ---- G2P Solver (gather grid to particles + advect) ----
class GPUAdvectorG2PSolver : public GPUModularSolver {
public:
    GPUAdvectorG2PSolver(GPUFluidContext& ctx,
                         std::shared_ptr<GPUAdvector> advImpl)
        : GPUModularSolver("gpu_advector_g2p"), impl_(std::move(advImpl))
    {
        (void)ctx;
    }

    void solve(GPUFluidContext& ctx, Real dt) override;

private:
    std::shared_ptr<GPUAdvector> impl_;
};

} // namespace fluid::gpu
