// ============================================================================
// include/FluidSim/fluid-types.h
// ============================================================================
#pragma once

#include <Core/core.h>
#include <Core/properties.h>
#include <Core/mesh.h>
#include <Core/aabb.h>
#include <array>
#include <optional>
#include <vector>

namespace fluid {

// Namespace alias: allows existing code using `core::Xxx` inside `namespace fluid`
// to resolve correctly to `ksk::core::Xxx`.
namespace core = ksk::core;

using Index = int;
using core::Real;
using core::NonCopyable;
using core::Vec2i;
using core::Vec3i;
using core::Vec3u;
using core::Vec2f;
using core::Vec3f;
using core::Vec2d;
using core::Vec3d;
using core::Vector;
using core::Matrix;
using core::Mesh;
using core::AABB;

// ============================== 枚举 ==============================

enum class AdvectionMethod { PIC, FLIP, APIC };

enum class PreconditionerMethod {
    None,
    Jacobi,                      // CPU/GPU
    ModifiedIncompleteCholesky,  // CPU only
    Multigrid,                   //
};

enum class ReconstructorMethod { Naive };

enum class CompressedSolverMethod {
    CG
};

enum class BoundaryType : uint8_t {
    FreeSlip,
    NoSlip,
    Inflow,
    Outflow,
    Periodic,
};


struct FluidDomain {
    Vec3d origin{0.0, 0.0, 0.0};
    Vec3d size{1.0, 1.0, 1.0};
    Vec3i resolution{64, 64, 64};
};

struct BoundaryConditions {
    std::array<BoundaryType, 6> types{};      // -X, +X, -Y, +Y, -Z, +Z
    std::array<Vec3d, 6> inflowVelocities{};  //
};

struct InitialFluid {
    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;  //
};

struct SolverConfig {
    AdvectionMethod advection = AdvectionMethod::FLIP;
    Real flipBlend = 0.97;

    PreconditionerMethod preconditioner = PreconditionerMethod::None;
    int pressureMaxIters = 300;
    Real pressureTolerance = 1e-4;

    Real density = 1000.0;
    Vec3d gravity{0.0, -9.8, 0.0};
    Real maxCfl = 5.0;
};

struct FluidScene {
    FluidDomain domain;
    BoundaryConditions boundaries;
    InitialFluid initialFluid;
    std::optional<Mesh> colliderMesh;
    SolverConfig solver;
};


struct FluidFrame {
    std::vector<Vec3d> particlePositions;
    std::vector<Vec3d> particleVelocities;
};

} // namespace fluid
