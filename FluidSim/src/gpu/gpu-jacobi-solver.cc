// ============================================================================
// src/gpu/gpu-jacobi-solver.cc
// ============================================================================

#include <FluidSim/gpu/gpu-jacobi-solver.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

using namespace sim::rhi;
GpuJacobiSolver::GpuJacobiSolver(Device& device, ShaderCompiler& compiler,
                                 const PressureSystem& system,
                                 const SolverConfig& config,
                                 const std::filesystem::path& shaderDir)
    : iterations_(config.pressureMaxIters)
{
    uint32_t nc = static_cast<uint32_t>(system.gridSize.x) *
                  system.gridSize.y * system.gridSize.z;

    pressurePing_ = device.createBuffer({
        .sizeBytes  = sizeof(float) * nc,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName  = "jacobi-ping",
    });

    psoJacobi_ = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "jacobi.hlsl");
}

void GpuJacobiSolver::solve(CommandList& cmd, const PressureSystem& system) {
    if (!psoJacobi_.valid()) return;

    uint32_t nc         = static_cast<uint32_t>(system.gridSize.x) *
                          system.gridSize.y * system.gridSize.z;
    uint32_t cellGroups = (nc + 255) / 256;

    // Clear both buffers
    cmd.fillBuffer(system.pressure, 0u);
    cmd.fillBuffer(pressurePing_,   0u);
    cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessTransferWrite,
                      BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

    // Ping-pong Jacobi
    // Even iterations: ping(in) → pressure(out)
    // Odd  iterations: pressure(in) → ping(out)
    int iters = std::max(1, iterations_);
    for (int i = 0; i < iters; ++i) {
        bool even = (i % 2 == 0);
        JacobiIterParams p;
        p.pressureIn   = even ? pressurePing_    : system.pressure;
        p.pressureOut  = even ? system.pressure  : pressurePing_;
        p.Adiag        = system.Adiag;
        p.Aneighbour0  = system.Aneighbour[0];
        p.Aneighbour1  = system.Aneighbour[1];
        p.Aneighbour2  = system.Aneighbour[2];
        p.Aneighbour3  = system.Aneighbour[3];
        p.Aneighbour4  = system.Aneighbour[4];
        p.Aneighbour5  = system.Aneighbour[5];
        p.rhs          = system.rhs;
        p.active       = system.active;
        p.gridSizeX    = static_cast<uint32_t>(system.gridSize.x);
        p.gridSizeY    = static_cast<uint32_t>(system.gridSize.y);
        p.gridSizeZ    = static_cast<uint32_t>(system.gridSize.z);
        p.numCells     = nc;
        cmd.dispatch(psoJacobi_, p, cellGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
    }

    // Ensure result is in system.pressure (true when iters is odd, even iters land in ping)
    if (iters % 2 == 0) {
        // Result is in pressurePing_; copy back to system.pressure
        std::array<BufferCopy, 1> region{{{0, 0, sizeof(float) * nc}}};
        cmd.copyBuffer(pressurePing_, system.pressure, region);
        cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                          BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderRead);
    }
}

} // namespace fluid::gpu
