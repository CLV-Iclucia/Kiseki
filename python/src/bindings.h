// Shared Python wrapper types for Kiseki bindings
// All binding files include this header to share type definitions.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <fem/system.h>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/constraints.h>
#include <fem/colliders.h>
#include <fem/ipc/integrator.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/fem-simulation.h>
#include <Deform/strain-energy-density.h>
#include <Maths/block-solvers/block-pcg.h>

#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <functional>
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace ksk::fem;
using namespace ksk::deform;

// ─── Python wrapper types ───────────────────────────────────────────────

/// Phase 3: Material wrapper
struct PyMaterial {
  std::unique_ptr<StrainEnergyDensity<Real>> impl;
  Real young;
  Real poisson;
};

/// Phase 7: Integrator config wrapper
struct PyIntegratorConfig {
  Real dHat = 1e-3;
  Real eps = 1e-2;
  Real kappa = 1e10;
  Real stepSizeScale = 0.9;
  bool locked{false};

  void check_locked() const {
    if (locked)
      throw std::runtime_error(
          "Integrator is locked after simulation has started. "
          "Cannot modify parameters after run() or step().");
  }
};

/// Phase 6: KinematicBody wrapper
struct PyKinematicBody {
  Collider body;
  std::string motion_type{"static"};
};

/// Phase 4: System wrapper with lock support
struct PySystem : std::enable_shared_from_this<PySystem> {
  System system;
  bool locked{false};

  // Per-primitive render color (indexed by add order); negative = use default
  std::vector<glm::vec3> primitive_colors;

  // Deferred constraints: stored until system.init() populates system.x
  std::vector<int> pending_pin_vertices;
  struct PrescribedMotionEntry {
    int vertex_idx;
    std::function<glm::dvec3(Real)> position_func;
  };
  std::vector<PrescribedMotionEntry> pending_prescribed_motions;

  void check_locked() const {
    if (locked)
      throw std::runtime_error(
          "System is locked after simulation has started. "
          "Cannot modify configuration after run() or step().");
  }

  void add_elastic_body(TetMesh& mesh, std::shared_ptr<PyMaterial> material, Real density,
                        glm::vec3 color = glm::vec3(-1.0f)) {
    check_locked();
    if (!material)
      throw py::type_error("material must be a valid NeoHookean object");
    if (density <= 0.0)
      throw py::value_error("density must be positive");

    auto energy = std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{material->young, material->poisson});

    ElasticTetMesh etm(std::move(mesh), std::move(energy), density);
    system.addPrimitive(Primitive(std::move(etm)));
    primitive_colors.push_back(color);
  }

  void add_kinematic_body(std::shared_ptr<PyKinematicBody> kb) {
    check_locked();
    if (!kb)
      throw py::type_error("kinematic_body must be a valid KinematicBody object");
    system.colliders().push_back(std::move(kb->body));
  }

  void set_gravity(py::array_t<double> g) {
    check_locked();
    auto buf = g.unchecked<1>();
    if (buf.shape(0) != 3)
      throw py::value_error("gravity must be a 3-element array");
    system.setGravity(glm::dvec3(buf(0), buf(1), buf(2)));
  }

  py::array_t<double> get_gravity() const {
    auto g = system.gravity();
    auto result = py::array_t<double>(3);
    auto buf = result.mutable_unchecked<1>();
    buf(0) = g.x; buf(1) = g.y; buf(2) = g.z;
    return result;
  }
};

/// Phase 5: Constraints accessor
/// Constraints are stored as "pending" operations until system.init() is called,
/// because system.x (the position vector) is empty before initialization.
struct PyConstraints {
  std::shared_ptr<PySystem> owner;

  void check_locked() const {
    owner->check_locked();
  }

  ConstraintManager& get_constraints() {
    return owner->system.constraints();
  }

  System& get_system() {
    return owner->system;
  }

  void pin_vertices(py::array_t<int> indices) {
    check_locked();
    auto buf = indices.unchecked<1>();
    for (py::ssize_t i = 0; i < buf.shape(0); i++) {
      if (buf(i) < 0)
        throw py::index_error("Vertex index must be non-negative, got " + std::to_string(buf(i)));
      owner->pending_pin_vertices.push_back(buf(i));
    }
  }

  void prescribe_motion(int vertex_idx,
                        std::function<py::array_t<double>(double)> position_func) {
    check_locked();
    if (vertex_idx < 0)
      throw py::index_error("Vertex index must be non-negative");

    auto cpp_pos_func = [position_func](Real t) -> glm::dvec3 {
      py::gil_scoped_acquire acquire;
      py::array_t<double> result = position_func(t);
      auto buf = result.unchecked<1>();
      if (buf.shape(0) != 3)
        throw std::runtime_error("Motion function must return a 3-element array");
      return glm::dvec3(buf(0), buf(1), buf(2));
    };

    owner->pending_prescribed_motions.push_back({vertex_idx, cpp_pos_func});
  }
};

/// Phase 8+9: Simulation wrapper with lock-after-run
struct PySimulation {
  std::shared_ptr<PySystem> py_system;
  std::shared_ptr<PyIntegratorConfig> py_integrator;
  std::unique_ptr<Integrator> integrator;
  bool initialized{false};
  bool has_run{false};
  int steps_completed{0};

  void ensure_initialized() {
    if (initialized) return;

    // 1. Initialize system (populates system.x with vertex positions)
    py_system->system.init();
    spdlog::info("[kiseki] system.init() done, x has {} blocks", py_system->system.x.numBlocks());

    // 2. Apply deferred constraints (pin_vertices needs system.x to be populated)
    if (!py_system->pending_pin_vertices.empty()) {
      spdlog::info("[kiseki] applying {} pending pin_vertices", py_system->pending_pin_vertices.size());
      py_system->system.constraints().pinVertices(
          py_system->pending_pin_vertices, py_system->system.x);
      py_system->pending_pin_vertices.clear();
    }
    for (auto& entry : py_system->pending_prescribed_motions) {
      py_system->system.constraints().prescribeMotion(
          entry.vertex_idx, entry.position_func);
    }
    py_system->pending_prescribed_motions.clear();

    // 3. Build constraint index (free/constrained DOF masks)
    if (!py_system->system.constraints().allConstraints().empty()) {
      spdlog::info("[kiseki] building constraints for {} blocks", py_system->system.x.numBlocks());
      py_system->system.constraints().build(py_system->system.x.numBlocks());
    }

    // 4. Create IPC integrator with solver
    spdlog::info("[kiseki] creating IPC integrator...");
    IpcIntegrator::Config cfg;
    cfg.dHat = py_integrator->dHat;
    cfg.eps = py_integrator->eps;
    cfg.contactStiffness = py_integrator->kappa;
    cfg.stepSizeScale = py_integrator->stepSizeScale;

    auto ipc = std::make_unique<IpcImplicitEuler>(py_system->system, cfg);
    spdlog::info("[kiseki] creating BlockPCG solver...");
    ipc->solver = std::make_unique<ksk::maths::BlockPCGSolver>(1000, 1e-6);
    integrator = std::move(ipc);

    initialized = true;
    spdlog::info("[kiseki] initialization complete");
  }

  void lock_all() {
    py_system->locked = true;
    py_integrator->locked = true;
    has_run = true;
  }

  void do_step(Real dt) {
    ensure_initialized();
    lock_all();
    spdlog::info("[kiseki] calling integrator->step(dt={})...", dt);
    integrator->step(dt);
    spdlog::info("[kiseki] step complete");
    // Note: IpcIntegrator::step() already calls advanceTime(dt) internally
    steps_completed++;
  }

  void run(Real dt, int steps) {
    for (int i = 0; i < steps; i++)
      do_step(dt);
  }
};

// ─── Binding registration declarations ──────────────────────────────────

void bind_mesh(py::module_& m);
void bind_material(py::module_& m);
void bind_system(py::module_& m);
void bind_constraints(py::module_& m);
void bind_kinematic(py::module_& m);
void bind_integrator(py::module_& m);
void bind_simulation(py::module_& m);
void bind_renderer(py::module_& m);
void bind_der(py::module_& m);
