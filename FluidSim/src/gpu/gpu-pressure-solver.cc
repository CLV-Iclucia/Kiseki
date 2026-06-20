// ============================================================================
// src/gpu/gpu-pressure-solver.cc
// Factory implementation for GpuPressureSolver::create()
// ============================================================================

#include <FluidSim/gpu/gpu-pressure-solver.h>
#include <FluidSim/gpu/gpu-jacobi-solver.h>
#include <FluidSim/gpu/gpu-pcg-solver.h>

namespace fluid::gpu {

std::unique_ptr<GpuPressureSolver> GpuPressureSolver::create(
    sim::rhi::Device&             device,
    sim::rhi::ShaderCompiler&     compiler,
    const PressureSystem&         system,
    const SolverConfig&           config,
    const std::filesystem::path&  shaderDir)
{
    switch (config.preconditioner) {
    case PreconditionerMethod::None:
        // "None" means no preconditioner → plain Jacobi iterations
        return std::make_unique<GpuJacobiSolver>(device, compiler, system, config, shaderDir);

    case PreconditionerMethod::Jacobi:
        // Jacobi used as CG preconditioner → full PCG
        return std::make_unique<GpuPCGSolver>(device, compiler, system, config, shaderDir);

    case PreconditionerMethod::ModifiedIncompleteCholesky:
        // MIC not available on GPU; fall back to PCG with Jacobi precond
        return std::make_unique<GpuPCGSolver>(device, compiler, system, config, shaderDir);

    case PreconditionerMethod::Multigrid:
        // Not yet implemented; fall back to PCG
        return std::make_unique<GpuPCGSolver>(device, compiler, system, config, shaderDir);
    }
    return std::make_unique<GpuPCGSolver>(device, compiler, system, config, shaderDir);
}

} // namespace fluid::gpu
