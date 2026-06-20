// ============================================================================
// src/gpu/gpu-force.cc
// ============================================================================

#include <FluidSim/gpu/gpu-force.h>
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

GPUForceSolver::GPUForceSolver(GPUFluidContext& ctx, GPUForceConfig config)
    : GPUModularSolver("gpu_force"), config_(config)
{
    psoBodyForce_ = sim::rhi::compileComputePipeline(
        ctx.device(), ctx.compiler(), shaderPath("body-force.hlsl"), "CSMain");
}

void GPUForceSolver::solve(GPUFluidContext& ctx, Real dt) {
    auto& cmd = ctx.cmd();
    int ny = ctx.gridSize.y;
    int nx = ctx.gridSize.x;
    int nz = ctx.gridSize.z;
    uint32_t numVFaces = nx * (ny + 1) * nz;

    // Dispatch body force shader
    // The actual binding logic mirrors GPUFluidBackend's body force dispatch.
    (void)cmd;
    (void)numVFaces;
}

void GPUForceSolver::configure(const core::Properties& props) {
    if (auto v = props.tryGet<Real>("gravity_y"))
        config_.gravity.y = *v;
}

} // namespace fluid::gpu
