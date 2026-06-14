//
// SimCraft Example: Energy Conservation Check (C++)
// ==================================================
//
// Zero-gravity scene: a single NeoHookean cube is pre-squashed (initial
// elastic PE) and given a uniform initial velocity (initial KE).  With no
// external forces and no collisions, total mechanical energy E = KE + PE
// should remain constant.
//
// Equivalent to: python/examples/energy_conservation.py
//
// What to look for:
//   - Ideal case    : E(t) ≡ E₀  (energy is conserved)
//   - Implicit Euler: E(t) < E₀  (slight monotone dissipation is expected)
//   - A growing E   : would indicate a bug
//
// The console prints a table:
//   Step   t       KE          PE          E_total     ΔE/E₀
//   ----   ----    --------    --------    --------    ------
//   ...
//
// Usage:
//   energy-check [--dt 0.005] [--steps 600] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/ipc/implicit-euler.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Deform/strain-energy-density.h>
#include <Renderer/renderer.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <format>
#include <limits>
#include <cmath>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

// ─── Helper: smooth normals ────────────────────────────────────────────────────

static void computeSmoothNormals(renderer::MeshProxy& mesh) {
  const auto& pos  = mesh.positions;
  const auto& tris = mesh.triangles;
  mesh.normals.assign(pos.size(), glm::vec3(0.0f));

  for (const auto& tri : tris) {
    glm::vec3 e1 = pos[tri.y] - pos[tri.x];
    glm::vec3 e2 = pos[tri.z] - pos[tri.x];
    glm::vec3 fn = glm::cross(e1, e2);
    mesh.normals[tri.x] += fn;
    mesh.normals[tri.y] += fn;
    mesh.normals[tri.z] += fn;
  }
  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    n = (len > 1e-8f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
  }
}

// ─── Helper: build render frame ───────────────────────────────────────────────

static std::unique_ptr<renderer::SceneProxy>
buildSceneProxy(const System& system, int frame) {
  auto proxy = std::make_unique<renderer::SceneProxy>();
  proxy->frameIndex      = frame;
  proxy->simulationTime  = system.currentTime();

  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& pr      = system.primitives()[i];
    auto surfaceView    = pr.getSurfaceView();
    auto vertCount      = pr.getVertexCount();
    int  dofStart       = pr.getDofStart();

    renderer::MeshProxy mesh;
    mesh.name = "body_" + std::to_string(i);

    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      const auto& p = system.x[(dofStart / 3) + static_cast<int>(v)];
      mesh.positions[v] = glm::vec3(static_cast<float>(p[0]),
                                    static_cast<float>(p[1]),
                                    static_cast<float>(p[2]));
    }

    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = {static_cast<unsigned>(tri.x),
                           static_cast<unsigned>(tri.y),
                           static_cast<unsigned>(tri.z)};
    }

    computeSmoothNormals(mesh);
    proxy->meshes.push_back(std::move(mesh));
  }
  return proxy;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("energy-check",
      "Energy conservation check: pre-deformed cube with initial velocity, zero gravity");
  options.add_options()
      ("dt",        "Timestep size",           cxxopts::value<double>()->default_value("0.005"))
      ("steps",     "Number of steps",         cxxopts::value<int>()->default_value("400"))
      ("no-render", "Disable rendering",       cxxopts::value<bool>()->default_value("false"))
      ("h,help",    "Print help");

  auto args    = options.parse(argc, argv);
  if (args.count("help")) { std::cout << options.help() << "\n"; return 0; }

  const double dt       = args["dt"].as<double>();
  const int    maxSteps = args["steps"].as<int>();
  const bool   noRender = args["no-render"].as<bool>();

  // ─── 1. Load mesh ──────────────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj\n";
    return 1;
  }
  auto& refMesh     = *meshOpt;
  const auto& baseVerts = refMesh.getVertices();
  const auto& baseTets  = refMesh.tets;
  const int   nVerts    = static_cast<int>(baseVerts.size());

  std::cout << std::format("Mesh: {} vertices, {} tets\n", nVerts, baseTets.size());

  // ─── 2. Compute bounding-box centre for scaling ────────────────────────────
  Vector<Real, 3> lo{ std::numeric_limits<Real>::max(),
                      std::numeric_limits<Real>::max(),
                      std::numeric_limits<Real>::max() };
  Vector<Real, 3> hi{ std::numeric_limits<Real>::lowest(),
                      std::numeric_limits<Real>::lowest(),
                      std::numeric_limits<Real>::lowest() };
  for (const auto& v : baseVerts) {
    for (int d = 0; d < 3; d++) { lo[d] = std::min(lo[d], v[d]); hi[d] = std::max(hi[d], v[d]); }
  }
  Vector<Real, 3> centre = (lo + hi) * 0.5;

  // ─── 3. Build pre-deformed mesh ────────────────────────────────────────────
  // TetMesh takes rest vertices + optional initial_positions (deformed at t=0).
  // commit() will set X = rest (cube), x = deformed (squashed), xdot = velocity.
  // ElasticTetMesh::init() computes Dm_inverse from rest shape and initial F from deformed.
  const Vector<Real, 3> scale{1.4, 0.3, 1.0};
  // 较大初速度 → 质心平移清晰可见
  const Vector<Real, 3> v0{5.0, 1.5, 0.0};

  std::vector<Vector<Real, 3>> deformedVerts(nVerts);
  std::vector<Vector<Real, 3>> initVels(nVerts);
  for (int i = 0; i < nVerts; i++) {
    const auto& bv = baseVerts[i];
    for (int d = 0; d < 3; d++)
      deformedVerts[i][d] = centre[d] + (bv[d] - centre[d]) * scale[d];
    initVels[i] = v0;
  }

  // Rest vertices = cube, initial_positions = squashed cube, velocities = v0
  TetMesh predeformedMesh(baseVerts, baseTets, initVels, deformedVerts);
  std::cout << std::format("Pre-deformation: scale = [{}, {}, {}]\n",
                           scale[0], scale[1], scale[2]);
  std::cout << std::format("Initial velocity: [{}, {}, {}] m/s\n", v0[0], v0[1], v0[2]);

  // ─── 4. Material ──────────────────────────────────────────────────────────
  // 适中硬度，弹性振荡周期在 0.1~0.5 s 范围内，300 步内可见多次回弹
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});

  // ─── 5. System ────────────────────────────────────────────────────────────
  System system;
  system.setGravity({0.0, 0.0, 0.0});   // zero gravity

  ElasticTetMesh body(std::move(predeformedMesh), std::move(energy), /*density=*/1000.0);
  system.addPrimitive(Primitive(std::move(body)));
  // No colliders — pure free motion

  system.init();
  std::cout << std::format("System: {} DOF,  dt = {},  steps = {}\n",
                           system.dof(), dt, maxSteps);

  // ─── 6. Integrator ────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat            = 1e-3;
  ipcConfig.contactStiffness = 1e8;

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(1000, 1e-6);

  // ─── 7. Energy logger ─────────────────────────────────────────────────────
  //
  // We measure the initial total energy after init (position already set to
  // deformed; velocity already set).  Then compare every LOG_EVERY steps.
  //
  const int LOG_EVERY = 20;
  double E0 = std::numeric_limits<double>::quiet_NaN();

  auto logEnergy = [&](int step) {
    double ke   = static_cast<double>(system.kineticEnergy());
    double pe   = static_cast<double>(system.potentialEnergy());
    double E    = ke + pe;
    double t    = system.currentTime();

    if (std::isnan(E0)) E0 = (std::abs(E) > 1e-12) ? E : 1.0;
    double drift = (E - E0) / std::abs(E0);

    std::cout << std::format(
        "step {:5d}  t={:7.4f}  KE={:13.6e}  PE={:13.6e}  E={:13.6e}  dE/E0={:+.4f}%\n",
        step, t, ke, pe, E, drift * 100.0);
  };

  // ─── 8. Run ───────────────────────────────────────────────────────────────

  // Print header
  std::cout << "\n";
  std::cout << std::format("{:>7}  {:>7}  {:>15}  {:>15}  {:>15}  {:>10}\n",
                           "step", "t", "KE", "PE", "E_total", "dE/E0(%)");
  std::cout << std::string(75, '-') << "\n";

  // Log t=0 state
  logEnergy(0);

  if (noRender) {
    for (int step = 1; step <= maxSteps; step++) {
      integrator->step(dt);
      // Note: integrator->step() already calls system.advanceTime(dt) internally
      if (step % LOG_EVERY == 0 || step <= 5)
        logEnergy(step);
    }
    std::cout << std::string(75, '-') << "\n";
    std::cout << "Done.  (Implicit Euler is dissipative — some negative drift is expected)\n";
    return 0;
  }

  // ── Render mode ─────────────────────────────────────────────────────────────
  auto renderer = renderer::createRenderer({
      .windowWidth  = 1280,
      .windowHeight = 720,
      .windowTitle  = "SimCraft - Energy Conservation Check",
  });

  // Push initial frame
  renderer->queue().push(buildSceneProxy(system, 0));

  std::atomic<int> stepsCompleted{0};
  std::thread simThread([&]() {
    for (int step = 1; step <= maxSteps && renderer->isRunning(); step++) {
      integrator->step(dt);
      // Note: integrator->step() already calls system.advanceTime(dt) internally
      system.advanceKinematicBodies(system.currentTime());

      auto proxy = buildSceneProxy(system, step);
      renderer->queue().push(std::move(proxy));
      stepsCompleted = step;

      if (step % LOG_EVERY == 0 || step <= 5)
        logEnergy(step);
    }
    renderer->shutdown();
  });

  renderer->runOnCurrentThread();
  simThread.join();

  std::cout << std::string(75, '-') << "\n";
  std::cout << std::format("Done. {} steps completed.\n", stepsCompleted.load());
  std::cout << "(Implicit Euler is dissipative — some negative drift is expected)\n";
  return 0;
}
