// ============================================================================
// FEM/src/cpu/cpu-fem-backend.cc
// CPUFEMBackend implementation — wraps existing System + IpcIntegrator
// ============================================================================

#include <fem/cpu/cpu-fem-backend.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/integrator.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/colliders.h>
#include <Deform/strain-energy-density.h>
#include <Maths/block-solvers/block-pcg.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace sim::fem {

// ============================================================================
// Helper: convert E, nu → mu, lambda (Lame parameters)
// ============================================================================
static std::pair<Real, Real> toLame(Real E, Real nu) {
    Real mu = E / (2.0 * (1.0 + nu));
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    return {mu, lambda};
}

// ============================================================================
// Helper: create constitutive model from string name + material params
// ============================================================================
static std::unique_ptr<deform::StrainEnergyDensity<Real>>
createConstitutiveModel(const std::string& name, Real E, Real nu) {
    auto [mu, lambda] = toLame(E, nu);

    if (name == "stable-neohookean" || name == "snh" || name == "StableNeoHookean") {
        return std::make_unique<deform::StableNeoHookean<Real>>(
            deform::ElasticityParameters<Real>{E, nu});
    }
    if (name == "arap" || name == "ARAP") {
        return std::make_unique<deform::ARAP<Real>>();
    }
    if (name == "linear" || name == "linear-elastic" || name == "LinearElastic") {
        return std::make_unique<deform::LinearElastic<Real>>(mu, lambda);
    }

    // Default: stable neo-hookean
    spdlog::warn("[CPUFEMBackend] Unknown constitutive model '{}', falling back to stable-neohookean", name);
    return std::make_unique<deform::StableNeoHookean<Real>>(
        deform::ElasticityParameters<Real>{E, nu});
}

// ============================================================================
// Helper: create MotionProfile from ColliderDesc
// ============================================================================
static std::unique_ptr<MotionProfile> createMotion(const FEMScene::ColliderDesc& desc) {
    if (desc.motionType == "constant_velocity") {
        return constantVelocity(desc.velocity);
    }
    if (desc.motionType == "sinusoidal") {
        return sinusoidalMotion(desc.axis, desc.amplitude, desc.frequency);
    }
    if (desc.motionType == "constant_rotation") {
        return constantRotation(desc.axis, desc.center, desc.omega);
    }
    return staticMotion();
}

// ============================================================================
// CPUFEMBackend::buildSystem
// ============================================================================
void CPUFEMBackend::buildSystem(const FEMScene& scene) {
    // --- Add primitives (ElasticTetMesh) ---
    for (const auto& meshDesc : scene.meshes) {
        // Convert glm::dvec3 → Eigen Vector<Real,3>
        std::vector<maths::Vector<Real, 3>> vertices(meshDesc.vertices.size());
        for (size_t i = 0; i < meshDesc.vertices.size(); i++) {
            vertices[i] = maths::Vector<Real, 3>(
                meshDesc.vertices[i].x, meshDesc.vertices[i].y, meshDesc.vertices[i].z);
        }

        // Convert initial positions
        std::vector<maths::Vector<Real, 3>> initialPositions;
        if (!meshDesc.initialPositions.empty()) {
            initialPositions.resize(meshDesc.initialPositions.size());
            for (size_t i = 0; i < meshDesc.initialPositions.size(); i++) {
                initialPositions[i] = maths::Vector<Real, 3>(
                    meshDesc.initialPositions[i].x, meshDesc.initialPositions[i].y, meshDesc.initialPositions[i].z);
            }
        }

        // Convert initial velocities
        std::vector<maths::Vector<Real, 3>> velocities;
        if (!meshDesc.initialVelocities.empty()) {
            velocities.resize(meshDesc.initialVelocities.size());
            for (size_t i = 0; i < meshDesc.initialVelocities.size(); i++) {
                velocities[i] = maths::Vector<Real, 3>(
                    meshDesc.initialVelocities[i].x, meshDesc.initialVelocities[i].y, meshDesc.initialVelocities[i].z);
            }
        }

        // Convert tet connectivity: array<int,4> → Tetrahedron (glm::ivec4)
        std::vector<Tetrahedron> tets(meshDesc.tets.size());
        for (size_t i = 0; i < meshDesc.tets.size(); i++) {
            tets[i] = Tetrahedron(
                meshDesc.tets[i][0], meshDesc.tets[i][1],
                meshDesc.tets[i][2], meshDesc.tets[i][3]);
        }

        // Build TetMesh
        TetMesh tetMesh(vertices, tets, velocities, initialPositions);

        // Create constitutive model
        auto energy = createConstitutiveModel(
            meshDesc.constitutiveModel, meshDesc.youngModulus, meshDesc.poissonRatio);

        // Build ElasticTetMesh
        ElasticTetMesh elasticMesh(std::move(tetMesh), std::move(energy), meshDesc.density);

        // Add as primitive
        system_.addPrimitive(Primitive(std::move(elasticMesh)));
    }

    // --- Add colliders ---
    for (const auto& colliderDesc : scene.colliders) {
        Collider collider;

        if (colliderDesc.mesh.has_value()) {
            auto triMesh = std::make_shared<TriangleMesh>();
            triMesh->vertices = colliderDesc.mesh->vertices;
            triMesh->triangles = colliderDesc.mesh->triangles;
            collider.geometry = Collider::MeshGeometry{triMesh};
        }

        collider.motion = createMotion(colliderDesc);
        system_.colliders().push_back(std::move(collider));
    }

    // --- Set gravity ---
    system_.setGravity(scene.gravity);

    // --- Initialize (builds mass matrix, geometry manager, etc.) ---
    system_.init();

    // --- Add constraints (after init so positions are available) ---
    int meshVertexOffset = 0;
    for (size_t meshIdx = 0; meshIdx < scene.meshes.size(); meshIdx++) {
        int meshVertCount = static_cast<int>(scene.meshes[meshIdx].vertices.size());

        for (const auto& cDesc : scene.constraints) {
            if (cDesc.meshIndex != static_cast<int>(meshIdx))
                continue;

            if (cDesc.type == "pin") {
                std::vector<int> globalIndices;
                for (int localIdx : cDesc.vertices) {
                    globalIndices.push_back(localIdx + meshVertexOffset);
                }
                system_.constraints().pinVertices(globalIndices, system_.x);
            } else if (cDesc.type == "prescribed_motion" && cDesc.motionType == "sinusoidal") {
                glm::dvec3 dir = glm::normalize(cDesc.axis);
                Real w = 2.0 * glm::pi<Real>() * cDesc.frequency;
                Real amp = cDesc.amplitude;

                for (int localIdx : cDesc.vertices) {
                    int globalIdx = localIdx + meshVertexOffset;
                    glm::dvec3 restPos = system_.x[globalIdx];

                    auto posFunc = [=](Real t) -> glm::dvec3 {
                        return restPos + dir * amp * std::sin(w * t);
                    };
                    auto velFunc = [=](Real t) -> glm::dvec3 {
                        return dir * amp * w * std::cos(w * t);
                    };
                    system_.constraints().prescribeMotion(globalIdx, posFunc, velFunc);
                }
            }
        }

        meshVertexOffset += meshVertCount;
    }

    // Rebuild constraint indices
    if (!system_.constraints().allConstraints().empty()) {
        system_.constraints().build(system_.x.numBlocks());
    }
}

// ============================================================================
// CPUFEMBackend::buildIntegrator
// ============================================================================
void CPUFEMBackend::buildIntegrator(const FEMScene& scene) {
    IpcIntegrator::Config cfg;
    cfg.dHat = scene.dHat;
    cfg.contactStiffness = scene.contactStiffness;
    cfg.eps = scene.convergenceEps;

    auto integrator = std::make_unique<IpcImplicitEuler>(system_, cfg);
    integrator->solver = makeLinearSolver(scene);

    // Track Newton iterations
    integrator->onNewtonIter = [this](int iter) {
        lastNewtonIters_ = iter;
    };

    integrator_ = std::move(integrator);
}

// ============================================================================
// CPUFEMBackend::makeLinearSolver — default CPU Block-Jacobi PCG
// ============================================================================
std::unique_ptr<maths::BlockLinearSolver>
CPUFEMBackend::makeLinearSolver(const FEMScene& scene) {
    return std::make_unique<maths::BlockPCGSolver>(
        scene.pcgMaxIter, scene.pcgTolerance);
}

// ============================================================================
// FEMBackend interface implementation
// ============================================================================

void CPUFEMBackend::initialize(const FEMScene& scene) {
    buildSystem(scene);
    buildIntegrator(scene);
    spdlog::info("[CPUFEMBackend] Initialized: {} vertices, {} tets",
                 system_.numVertices(), 
                 [&]() { 
                     int total = 0;
                     for (const auto& m : scene.meshes) total += static_cast<int>(m.tets.size());
                     return total;
                 }());
}

void CPUFEMBackend::step(Real dt) {
    lastNewtonIters_ = 0;
    integrator_->step(dt);
}

void CPUFEMBackend::readback(FEMFrame& out) {
    int nBlocks = system_.x.numBlocks();

    out.positions.resize(nBlocks);
    out.velocities.resize(nBlocks);

    for (int i = 0; i < nBlocks; i++) {
        out.positions[i] = system_.x[i];
        out.velocities[i] = system_.xdot[i];
    }

    out.kineticEnergy = system_.kineticEnergy();
    out.potentialEnergy = system_.potentialEnergy();
    out.totalEnergy = out.kineticEnergy + out.potentialEnergy;
    out.time = system_.currentTime();
    out.newtonIters = lastNewtonIters_;
}

// ============================================================================
// Factory function
// ============================================================================

std::unique_ptr<FEMBackend> createFEMBackend(const std::string& type) {
    if (type == "cpu") {
        return std::make_unique<CPUFEMBackend>();
    }
    // GPU backend will be added in Phase 2
    throw std::runtime_error("[createFEMBackend] Unknown backend type: " + type);
}

} // namespace sim::fem
