//
// SimCraft Example: Multi-Body Collision (C++)
// =============================================
//
// Two cubes fly toward each other in zero gravity and collide.
// Equivalent to: python/examples/multi_body.py
//
// Demonstrates:
//   - Multiple elastic bodies with initial velocities
//   - Zero gravity (pure momentum exchange)
//   - Elastic-elastic collision via IPC
//   - SimulationApp API (no manual frame pushing needed)
//
// Usage:
//   multi-body [--dt 0.005] [--steps 500] [--no-render]
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
#include <fem/scene-proxy.h>
#include <iostream>
#include <format>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("multi-body", "Two cubes head-on collision with IPC");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.005"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("500"))
      ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const double dt = args["dt"].as<double>();
  const int maxSteps = args["steps"].as<int>();
  const bool noRender = args["no-render"].as<bool>();

  // ─── 1. Load template mesh ──────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj" << std::endl;
    return 1;
  }
  auto templateMesh = std::move(*meshOpt);
  const auto& baseVerts = templateMesh.getVertices();
  const auto& baseTets = templateMesh.tets;
  const int nVerts = static_cast<int>(baseVerts.size());

  // Compute mesh width for spacing
  Real xMin = std::numeric_limits<Real>::max();
  Real xMax = std::numeric_limits<Real>::lowest();
  for (const auto& v : baseVerts) {
    xMin = std::min(xMin, v[0]);
    xMax = std::max(xMax, v[0]);
  }
  Real width = xMax - xMin;
  Real separation = 0.05;  // initial gap
  Real speed = 2.0;        // m/s

  std::cout << std::format("Template mesh: {} verts, {} tets, width = {:.4f}\n",
                           nVerts, baseTets.size(), width);

  // ─── 2. Material ────────────────────────────────────────────────────────────
  auto makeEnergy = []() {
    return std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{.E = 2e5, .nu = 0.4});
  };

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;
  system.setGravity({0.0, 0.0, 0.0});  // zero gravity

  // Left cube: offset left, velocity → right (+x)
  {
    std::vector<Vector<Real, 3>> verts(nVerts);
    std::vector<Vector<Real, 3>> vels(nVerts);
    Real offsetX = -(width / 2.0 + separation / 2.0);
    for (int i = 0; i < nVerts; i++) {
      verts[i] = baseVerts[i];
      verts[i][0] += offsetX;
      vels[i] = Vector<Real, 3>(speed, 0.0, 0.0);
    }
    TetMesh leftMesh(verts, baseTets, vels);
    ElasticTetMesh leftBody(std::move(leftMesh), makeEnergy(), /*density=*/800.0);
    system.addPrimitive(Primitive(std::move(leftBody)));
    std::cout << std::format("  Left cube:  {} verts, vel = [+{}, 0, 0]\n", nVerts, speed);
  }

  // Right cube: offset right, velocity → left (-x)
  {
    std::vector<Vector<Real, 3>> verts(nVerts);
    std::vector<Vector<Real, 3>> vels(nVerts);
    Real offsetX = +(width / 2.0 + separation / 2.0);
    for (int i = 0; i < nVerts; i++) {
      verts[i] = baseVerts[i];
      verts[i][0] += offsetX;
      vels[i] = Vector<Real, 3>(-speed, 0.0, 0.0);
    }
    TetMesh rightMesh(verts, baseTets, vels);
    ElasticTetMesh rightBody(std::move(rightMesh), makeEnergy(), /*density=*/800.0);
    system.addPrimitive(Primitive(std::move(rightBody)));
    std::cout << std::format("  Right cube: {} verts, vel = [-{}, 0, 0]\n", nVerts, speed);
  }

  // 初始化系统
  system.init();
  std::cout << std::format("System: {} DOF, {} bodies\n", system.dof(), system.primitives().size());

  // ─── 4. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 2e-3;
  ipcConfig.contactStiffness = 1e9;  // kappa

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(1000, 1e-6);

  std::cout << std::format("IPC: dHat = {}, kappa = {}, dt = {}\n",
                           ipcConfig.dHat, ipcConfig.contactStiffness, dt);

  // ─── 5. Run ─────────────────────────────────────────────────────────────────

  renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Multi-Body Collision",
  });

  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(system, step);
  };

  app.logInterval = 50;
  app.logFn = [&](int step) {
    std::cout << std::format("Step {:4d}, t = {:.4f}\n", step, system.currentTime());
  };

  int completed;
  if (noRender) {
    completed = app.runHeadless(maxSteps);
  } else {
    completed = app.run(maxSteps);
  }

  std::cout << std::format("Done. {} steps completed.\n", completed);
  return 0;
}
