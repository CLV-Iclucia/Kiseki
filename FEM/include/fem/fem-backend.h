// ============================================================================
// FEM/include/fem/fem-backend.h
// FEMBackend thin interface + FEMScene (input) + FEMFrame (output)
// ============================================================================
#pragma once

#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>

namespace sim::fem {

using Real = double;

// ============================================================================
// FEMScene — pure POD scene description (CPU/GPU agnostic)
// ============================================================================

struct FEMScene {
    // ---- Tet mesh descriptions ----
    struct TetMeshDesc {
        std::vector<glm::dvec3> vertices;              // rest configuration (X)
        std::vector<glm::dvec3> initialPositions;      // initial deformed config (x at t=0); empty → use vertices
        std::vector<glm::dvec3> initialVelocities;     // initial velocity; empty → zero
        std::vector<std::array<int, 4>> tets;          // tetrahedra connectivity
        std::vector<std::array<int, 3>> triangles;     // surface triangles
        std::vector<std::array<int, 2>> edges;         // surface edges

        // Material
        std::string constitutiveModel = "stable-neohookean";
        Real youngModulus = 1e6;
        Real poissonRatio = 0.45;
        Real density = 1000.0;
    };
    std::vector<TetMeshDesc> meshes;

    // ---- Colliders (kinematic bodies) ----
    struct ColliderDesc {
        struct MeshData {
            std::vector<glm::dvec3> vertices;
            std::vector<glm::ivec3> triangles;
        };
        std::optional<MeshData> mesh;

        // Motion type
        std::string motionType = "static";  // "static", "constant_velocity", "sinusoidal", "constant_rotation"
        glm::dvec3 velocity{0.0};
        glm::dvec3 axis{0, 1, 0};
        glm::dvec3 center{0.0};
        Real amplitude = 0.0;
        Real frequency = 0.0;
        Real omega = 0.0;
    };
    std::vector<ColliderDesc> colliders;

    // ---- Constraints ----
    struct ConstraintDesc {
        std::string type;                      // "pin", "prescribed_motion"
        std::vector<int> vertices;             // vertex indices (per-mesh local)
        int meshIndex = 0;                     // which mesh this constraint belongs to
        glm::bvec3 mask{true, true, true};     // which components are constrained

        // For prescribed_motion
        std::string motionType;
        glm::dvec3 axis{0.0};
        Real amplitude = 0.0;
        Real frequency = 0.0;
    };
    std::vector<ConstraintDesc> constraints;

    // ---- Global parameters ----
    glm::dvec3 gravity{0, -9.81, 0};

    // ---- IPC parameters ----
    Real dHat = 1e-3;
    Real contactStiffness = 1e10;
    Real convergenceEps = 1e-2;

    // ---- Solver parameters ----
    int pcgMaxIter = 1000;
    Real pcgTolerance = 1e-6;
};

// ============================================================================
// FEMFrame — simulation output per frame
// ============================================================================

struct FEMFrame {
    std::vector<glm::dvec3> positions;      // current vertex positions (all meshes concatenated)
    std::vector<glm::dvec3> velocities;     // current velocities
    Real totalEnergy = 0;
    Real kineticEnergy = 0;
    Real potentialEnergy = 0;
    Real time = 0;
    int newtonIters = 0;
    int pcgIters = 0;
};

// ============================================================================
// FEMBackend — thin interface
// ============================================================================

struct FEMBackend {
    virtual ~FEMBackend() = default;

    /// Initialize from a scene description
    virtual void initialize(const FEMScene& scene) = 0;

    /// Advance one time step
    virtual void step(Real dt) = 0;

    /// Read back simulation results into a frame
    virtual void readback(FEMFrame& out) = 0;
};

/// Factory: create backend by name ("cpu" or "gpu")
std::unique_ptr<FEMBackend> createFEMBackend(const std::string& type);

} // namespace sim::fem
