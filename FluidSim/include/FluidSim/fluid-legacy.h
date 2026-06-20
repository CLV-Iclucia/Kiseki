// ============================================================================
// include/FluidSim/fluid-legacy.h
// Legacy interfaces — to be removed after migration is complete.
// Kept temporarily so existing code (cpu::FluidSimulator, CPUFluidBackend,
// GPUFluidBackend) still compiles during the transition.
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <Core/animation.h>
#include <Core/error.h>
#include <Core/mesh.h>
#include <memory>
#include <utility>

namespace fluid {

// ---- Legacy enums (migrated from old fluid-simulator.h) ----

enum class ProjectSolver {
    FVM,
    FDM
};

enum class Backend {
    CPU,
    CUDA
};

// ---- Legacy interface (to be removed) ----

struct FluidComputeBackend : NonCopyable {
    virtual void setCollider(const Mesh& collider_mesh) const = 0;
    virtual void setInitialFluid(const Mesh& fluid_mesh) = 0;
    virtual void setAdvector(AdvectionMethod advection_method) = 0;
    virtual void setProjector(ProjectSolver project_solver) = 0;
    virtual void setCompressedSolver(CompressedSolverMethod solver_method,
                                     PreconditionerMethod preconditioner_method) = 0;
    virtual void setReconstructor(ReconstructorMethod reconstructor_method) = 0;
    virtual ~FluidComputeBackend() = default;
};

struct Scene {
    Mesh collider_mesh;
    Mesh fluid_init_mesh;
};

struct SimConfig {
    int nParticles = 0;
    Vec3d size;
    Vec3d orig;
    Vec3i resolution;
};

} // namespace fluid
