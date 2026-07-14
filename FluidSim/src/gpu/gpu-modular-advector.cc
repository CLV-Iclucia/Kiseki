// ============================================================================
// src/gpu/gpu-modular-advector.cc
// ============================================================================

#include <FluidSim/gpu/gpu-modular-advector.h>
#include <FluidSim/gpu/gpu-backend.h>
#include <Core/profiler.h>

namespace fluid::gpu {

// ---- Helper: build a GPUGridState view from GPUFluidContext ----
static GPUGridState makeGridView(GPUFluidContext& ctx) {
    GPUGridState g;
    g.uGrid = ctx.uGrid;
    g.vGrid = ctx.vGrid;
    g.wGrid = ctx.wGrid;
    g.uGridBuf = ctx.uGridBuf;
    g.vGridBuf = ctx.vGridBuf;
    g.wGridBuf = ctx.wGridBuf;
    g.uValid = ctx.uValid;
    g.vValid = ctx.vValid;
    g.wValid = ctx.wValid;
    g.uValidBuf = ctx.uValidBuf;
    g.vValidBuf = ctx.vValidBuf;
    g.wValidBuf = ctx.wValidBuf;
    g.fluidSdfImg = ctx.fluidSdfImg;
    g.colliderSdfImg = ctx.colliderSdfImg;
    g.sdfSampler = ctx.sdfSampler;
    g.particlePositions = ctx.particlePositions;
    g.particleVelocities = ctx.particleVelocities;
    g.gridSize = ctx.gridSize;
    g.gridSpacing = ctx.gpuGridSpacing;
    g.originX = ctx.originX;
    g.originY = ctx.originY;
    g.originZ = ctx.originZ;
    g.numParticles = ctx.numParticles;
    return g;
}

void GPUAdvectorP2GSolver::solve(GPUFluidContext& ctx, Real dt) {
    SIM_PROFILE_FUNCTION();

    GPUGridState grid = makeGridView(ctx);
    impl_->scatterP2G(ctx.cmd(), grid, dt);
}

void GPUAdvectorG2PSolver::solve(GPUFluidContext& ctx, Real dt) {
    SIM_PROFILE_FUNCTION();

    GPUGridState grid = makeGridView(ctx);
    impl_->gatherAndAdvect(ctx.cmd(), grid, dt);
}

} // namespace fluid::gpu
