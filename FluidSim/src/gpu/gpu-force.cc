// ============================================================================
// src/gpu/gpu-force.cc
// ============================================================================

#include <FluidSim/gpu/gpu-force.h>

namespace fluid::gpu {

using namespace ksk::rhi;

GPUForceSolver::GPUForceSolver(GPUFluidContext& ctx, GPUForceConfig config)
    : GPUModularSolver("gpu_force"),
      config_(config),
      bodyForce_(ctx.device())
{}

void GPUForceSolver::solve(GPUFluidContext& ctx, Real dt) {
    auto& cmd = ctx.cmd();
    int ny = ctx.gridSize.y;
    int nx = ctx.gridSize.x;
    int nz = ctx.gridSize.z;
    uint32_t numVFaces = nx * (ny + 1) * nz;

    if (!bodyForce_) return;

    BodyForceCS::Params params;
    params.vGrid = ctx.vGrid;
    params.dt = static_cast<float>(dt);
    params.gravityY = static_cast<float>(config_.gravity.y);
    params.numVFaces = numVFaces;
    cmd.dispatch(bodyForce_, params, (numVFaces + 255) / 256, 1, 1);
    cmd.memoryBarrier(BarrierDesc::StageComputeShader,
                      BarrierDesc::StageComputeShader);
}

void GPUForceSolver::configure(const core::Properties& props) {
    if (auto v = props.tryGet<Real>("gravity_y"))
        config_.gravity.y = *v;
}

} // namespace fluid::gpu
