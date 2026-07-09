// ============================================================================
// src/gpu/gpu-pressure-solver.cc
// Factory implementation for GpuPressureSolver::create()
// ============================================================================

#include <FluidSim/gpu/gpu-pressure-solver.h>
#include <FluidSim/gpu/gpu-jacobi-solver.h>
#include <FluidSim/gpu/gpu-pcg-solver.h>

namespace fluid::gpu {

std::unique_ptr<GPUPressureSolver> GPUPressureSolver::create(
    sim::rhi::Device& device,
    const PressureSystem& system,
    const SolverConfig& config)
{
    switch (config.preconditioner) {
    case PreconditionerMethod::None:
        // "None" means no preconditioner → plain Jacobi iterations
        return std::make_unique<GPUJacobiSolver>(device, system, config);

    case PreconditionerMethod::Jacobi:
        // Jacobi used as CG preconditioner → full PCG
        return std::make_unique<GpuPCGSolver>(device, system, config);

    case PreconditionerMethod::ModifiedIncompleteCholesky:
        // MIC not available on GPU; fall back to PCG with Jacobi precond
        return std::make_unique<GpuPCGSolver>(device, system, config);

    case PreconditionerMethod::Multigrid:
        // Not yet implemented; fall back to PCG
        return std::make_unique<GpuPCGSolver>(device, system, config);
    }
    return std::make_unique<GpuPCGSolver>(device, system, config);
}

} // namespace fluid::gpu
