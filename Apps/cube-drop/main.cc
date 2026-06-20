//
// SimCraft Example: Cube Free Fall (C++)
// =======================================
//
// A cube mesh falls under gravity and collides with a ground plane.
// Equivalent to: python/examples/cube_drop.py
//
// Demonstrates:
//   TetMesh loading → material creation → System setup →
//   kinematic ground plane → IPC integrator → realtime rendering.
//
// Usage:
//   cube-drop [--dt 0.01] [--steps 500] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/colliders.h>
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
  cxxopts::Options options("cube-drop", "Cube free-fall with IPC collision");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
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

  // ─── 1. Mesh ────────────────────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj" << std::endl;
    return 1;
  }
  auto tetMesh = std::move(*meshOpt);
  std::cout << std::format("Mesh: {} vertices, {} tets\n",
                           tetMesh.getVertices().size(), tetMesh.tets.size());

  // ─── 2. Material ────────────────────────────────────────────────────────────
  // NeoHookean: Young's modulus = 1e5, Poisson's ratio = 0.4
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;

  // 弹性体 = mesh + material + density
  ElasticTetMesh elasticBody(std::move(tetMesh), std::move(energy), /*density=*/1000.0);
  system.addPrimitive(Primitive(std::move(elasticBody)));

  // 重力
  system.setGravity({0.0, -9.81, 0.0});

  // ─── 4. Ground Plane ────────────────────────────────────────────────────────
  // y = -1 平面 (法线朝上, offset = -1)
  {
    Collider ground;
    glm::dvec3 normal(0.0, 1.0, 0.0);
    double offset = -1.0;

    Collider::SDFGeometry sdf;
    sdf.signedDistance = [normal, offset](const glm::dvec3& p) -> Real {
      return glm::dot(normal, p) - offset;
    };
    sdf.gradient = [normal](const glm::dvec3&) -> glm::dvec3 {
      return normal;
    };

    ground.geometry = std::move(sdf);
    ground.motion = staticMotion();
    system.colliders().push_back(std::move(ground));
  }

  // 初始化系统（分配 DOF、构建质量矩阵等）
  system.init();

  // ─── 5. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 1e-3;
  ipcConfig.contactStiffness = 1e8;  // kappa

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(/*maxIter=*/1000, /*tol=*/1e-6);

  std::cout << std::format("System: {} DOF, dt = {}, max_steps = {}\n",
                           system.dof(), dt, maxSteps);
  std::cout << std::format("IPC: dHat = {}, kappa = {}\n",
                           ipcConfig.dHat, ipcConfig.contactStiffness);

  // ─── 6. Run ─────────────────────────────────────────────────────────────────

  renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Cube Drop",
  });

  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(system, step);
  };

  app.logInterval = 50;
  app.logFn = [&](int step) {
    std::cout << std::format("Step {:4d}, t = {:.4f}, E = {:.6f}\n",
                             step, system.currentTime(), system.totalEnergy());
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
