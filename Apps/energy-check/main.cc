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
#include <Renderer/simulation-app.h>
#include <Renderer/fem-scene-proxy.h>
#include <iostream>
#include <format>
#include <limits>
#include <cmath>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

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
  const Vector<Real, 3> scale{1.4, 0.3, 1.0};
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
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});

  // ─── 5. System ────────────────────────────────────────────────────────────
  System system;
  system.setGravity({0.0, 0.0, 0.0});   // zero gravity

  ElasticTetMesh body(std::move(predeformedMesh), std::move(energy), /*density=*/1000.0);
  system.addPrimitive(Primitive(std::move(body)));

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

  renderer::SimulationApp app({
      .windowWidth  = 1280,
      .windowHeight = 720,
      .windowTitle  = "SimCraft - Energy Conservation Check",
  });

  // 注意 energy-check 的步数从 1 开始（step=0 已经是初始状态）
  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(system, step);
  };

  // 使用自定义 logFn 输出能量表
  app.logInterval = LOG_EVERY;
  app.logFn = [&](int step) {
    logEnergy(step + 1); // step 在 SimulationApp 中从 0 开始，对应模拟的第 step+1 步
  };

  int completed;
  if (noRender) {
    completed = app.runHeadless(maxSteps);
  } else {
    completed = app.run(maxSteps);
  }

  std::cout << std::string(75, '-') << "\n";
  std::cout << std::format("Done. {} steps completed.\n", completed);
  std::cout << "(Implicit Euler is dissipative — some negative drift is expected)\n";
  return 0;
}
