// ============================================================================
// include/FluidSim/gpu/gpu-pressure-solver.h
//
// Abstract interface for GPU pressure solvers.
//
// Design rationale:
//   GPUProjector owns the system-build and velocity-projection steps (which
//   are fixed regardless of solver choice).  The *linear solve* step
//   (Ap = b) is delegated to a GpuPressureSolver via this interface.
//
// Implementations:
//   GpuJacobiSolver   — simple Jacobi iterations (ping-pong); easy to debug
//   GpuPCGSolver      — Jacobi-preconditioned CG; better convergence
//   (future) GpuGaussSeidelSolver, GpuMultigridSolver, ...
//
// A concrete solver is created via GpuPressureSolver::create() based on
// the PreconditionerMethod in SolverConfig.
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>   // GPUGridState
#include <FluidSim/fluid-types.h>        // SolverConfig, PreconditionerMethod
#include <RHI/rhi.h>
#include <filesystem>
#include <memory>

namespace fluid::gpu {

// ---- Buffers shared between projector and solver ----
// The projector owns these; the solver receives them each call.
struct PressureSystem {
    sim::rhi::BufferRef Adiag;
    sim::rhi::BufferRef Aneighbour[6];  // -X +X -Y +Y -Z +Z
    sim::rhi::BufferRef rhs;
    sim::rhi::BufferRef active;
    sim::rhi::BufferRef pressure;       // output (read/write)
    Vec3i               gridSize;
};

// ============================================================================
// Abstract solver interface
// ============================================================================
class GPUPressureSolver {
public:
    virtual ~GPUPressureSolver() = default;

    // Solve Ap = b, writing result into system.pressure.
    // The system matrices are already filled by GPUProjector::buildWeightsAndSystem.
    virtual void solve(sim::rhi::CommandList& cmd,
                       const PressureSystem& system) = 0;

    // Update iteration count / tolerance at runtime (no GPU resource realloc).
    virtual void updateConfig(const SolverConfig& config) = 0;

    // Factory: create solver based on SolverConfig::preconditioner.
    // Jacobi → GpuJacobiSolver
    // None / Jacobi (as PCG preconditioner) → GpuPCGSolver
    static std::unique_ptr<GPUPressureSolver> create(
        sim::rhi::Device& device,
        const PressureSystem& system,
        const SolverConfig& config);
};

} // namespace fluid::gpu
