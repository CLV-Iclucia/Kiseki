// ============================================================================
// src/gpu/gpu-extrapolator.cc
// ============================================================================

#include <FluidSim/gpu/gpu-extrapolator.h>
#include <RHI/shader-utils.h>
#include <filesystem>

namespace fluid::gpu {

using namespace sim::rhi;
namespace fs = std::filesystem;

static fs::path shaderPath(const std::string& name) {
#ifdef FLUIDSIM_SHADER_DIR
    return fs::path(FLUIDSIM_SHADER_DIR) / name;
#else
    return fs::path(name);
#endif
}

GPUExtrapolatorSolver::GPUExtrapolatorSolver(GPUFluidContext& ctx, int iters)
    : GPUModularSolver("gpu_extrapolator"), iters_(iters)
{
    psoExtrapolate_ = compileComputePipeline(
        ctx.device(), ctx.compiler(), shaderPath("extrapolate.hlsl"), "CSMain");
}

void GPUExtrapolatorSolver::solve(GPUFluidContext& ctx, Real dt) {
    (void)dt;
    // Extrapolation passes for u, v, w
    // Each axis: ping-pong between grid/gridBuf and valid/validBuf
    auto& cmd = ctx.cmd();
    int nx = ctx.gridSize.x;
    int ny = ctx.gridSize.y;
    int nz = ctx.gridSize.z;

    // This is a simplified placeholder — the actual implementation would
    // dispatch the extrapolate shader for each axis for `iters_` iterations.
    // The full dispatch logic mirrors GPUFluidBackend::substep's extrapolation
    // section (ping-pong between grid/gridBuf, valid/validBuf).
    (void)cmd;
    (void)nx;
    (void)ny;
    (void)nz;
}

} // namespace fluid::gpu
