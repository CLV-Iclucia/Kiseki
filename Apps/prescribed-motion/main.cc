//
// SimCraft Example: Prescribed Motion (C++)
// ==========================================
//
// A cube mesh has one end pinned and one vertex driven by sinusoidal motion.
// Equivalent to: python/examples/prescribed_motion.py
//
// Demonstrates:
//   TetMesh loading → material → System → pin constraints →
//   prescribed time-varying motion → IPC integrator → realtime rendering.
//
// Usage:
//   prescribed-motion [--dt 0.01] [--steps 300] [--no-render]
//

#include <cxxopts.hpp>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/constraints.h>
#include <fem/ipc/implicit-euler.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Deform/strain-energy-density.h>
#include <Renderer/simulation-app.h>
#include <fem/scene-proxy.h>
#include <iostream>
#include <cmath>
#include <format>
#include <numbers>

using namespace sim;
using namespace sim::fem;
using namespace sim::deform;

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("prescribed-motion",
      "Elastic body with pinned + prescribed sinusoidal motion");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("300"))
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
  const auto& verts = tetMesh.getVertices();
  std::cout << std::format("Mesh: {} vertices, {} tets\n", verts.size(), tetMesh.tets.size());

  // Find x_min face (pinned end) and x_max face (driven end)
  Real xMin = std::numeric_limits<Real>::max();
  Real xMax = std::numeric_limits<Real>::lowest();
  for (const auto& v : verts) {
    xMin = std::min(xMin, v.x());
    xMax = std::max(xMax, v.x());
  }
  Real eps = (xMax - xMin) * 1e-6;

  std::vector<int> pinnedFace, drivenFace;
  for (int i = 0; i < static_cast<int>(verts.size()); i++) {
    if (std::abs(verts[i].x() - xMin) < eps) pinnedFace.push_back(i);
    if (std::abs(verts[i].x() - xMax) < eps) drivenFace.push_back(i);
  }

  // ─── 2. Material ────────────────────────────────────────────────────────────
  // NeoHookean: Young's = 2e5, Poisson's = 0.45
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 2e5, .nu = 0.45});

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;

  ElasticTetMesh elasticBody(std::move(tetMesh), std::move(energy), /*density=*/800.0);
  system.addPrimitive(Primitive(std::move(elasticBody)));

  system.setGravity({0.0, -9.81, 0.0});

  // 初始化系统（分配 DOF）
  system.init();

  // ─── 4. Constraints ─────────────────────────────────────────────────────────
  // Pin the entire x_min face
  system.constraints().pinVertices(pinnedFace, system.x);
  std::cout << std::format("Pinned {} vertices on x_min face\n", pinnedFace.size());

  // Prescribe sinusoidal y-motion on the entire x_max face
  constexpr double amplitude = 0.3;
  constexpr double freq = 1.0;  // Hz

  for (int vIdx : drivenFace) {
    system.constraints().prescribeMotion(vIdx,
        [amplitude, freq](Real t) -> glm::dvec3 {
          return glm::dvec3(
              0.0,
              amplitude * std::sin(2.0 * std::numbers::pi * freq * t),
              0.0);
        });
  }
  std::cout << std::format("Prescribed sinusoidal motion on {} vertices (x_max face)\n",
                           drivenFace.size());

  // Build constraint index
  system.constraints().build(system.x.numBlocks());

  // ─── 5. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 1e-3;
  ipcConfig.contactStiffness = 1e8;

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(1000, 1e-6);

  std::cout << std::format("System: {} DOF, dt = {}, max_steps = {}\n",
                           system.dof(), dt, maxSteps);

  // ─── 6. Run ─────────────────────────────────────────────────────────────────

  // 物体颜色：青色 (同 Python 例子)
  glm::vec3 color(0.35f, 0.80f, 0.80f);

  renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Prescribed Motion",
  });

  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  // 使用自定义 buildProxy 以设置 per-object 颜色
  app.buildProxy = [&](int step) {
    auto proxy = renderer::buildSceneProxyFromSystem(system, step);
    // 设置自定义物体颜色
    for (auto& mesh : proxy->meshes) {
      mesh.objectColor = color;
    }
    return proxy;
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
