//
// SimCraft Example: Mesh Collider (C++)
// ======================================
//
// A cube mesh falls under gravity onto a triangulated ground plane (MeshGeometry collider).
// This demonstrates the unified IPC collision system where both deformable
// and kinematic mesh geometry are handled through GeometryManager.
//
// Key differences from cube-drop:
//   - Ground is a **triangle mesh** collider (not SDF)
//   - The unified collision pipeline detects deformable-vs-kinematic contacts
//     using the same BVH and constraint code path as deformable-vs-deformable
//
// Usage:
//   mesh-collider [--dt 0.01] [--steps 500] [--no-render]
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

using namespace ksk;
using namespace ksk::fem;
using namespace ksk::deform;

// ─── Helper: 创建地面三角网格 ────────────────────────────────────────────────
// 在 y = groundY 处创建一个 gridN x gridN 的正方形平面，中心在原点。
static std::shared_ptr<TriangleMesh> createGroundMesh(
    Real halfExtent = 3.0, Real groundY = -1.0, int gridN = 4) {
  auto mesh = std::make_shared<TriangleMesh>();

  // 顶点：(gridN+1) x (gridN+1) 的网格
  int nSide = gridN + 1;
  mesh->vertices.reserve(nSide * nSide);
  for (int j = 0; j < nSide; ++j) {
    for (int i = 0; i < nSide; ++i) {
      Real u = static_cast<Real>(i) / gridN;  // [0, 1]
      Real v = static_cast<Real>(j) / gridN;
      Real x = (u - 0.5) * 2.0 * halfExtent;
      Real z = (v - 0.5) * 2.0 * halfExtent;
      mesh->vertices.emplace_back(x, groundY, z);
    }
  }

  // 三角形：每个格子 2 个三角形
  mesh->triangles.reserve(gridN * gridN * 2);
  for (int j = 0; j < gridN; ++j) {
    for (int i = 0; i < gridN; ++i) {
      int v00 = j * nSide + i;
      int v10 = j * nSide + (i + 1);
      int v01 = (j + 1) * nSide + i;
      int v11 = (j + 1) * nSide + (i + 1);
      mesh->triangles.emplace_back(v00, v10, v11);
      mesh->triangles.emplace_back(v00, v11, v01);
    }
  }

  // 提取边集
  mesh->computeEdges();

  return mesh;
}

// ─── Helper: 创建倾斜斜面三角网格 ───────────────────────────────────────────
// 一个沿 x 方向倾斜的平面，用于展示碰撞体上的滑动行为。
static std::shared_ptr<TriangleMesh> createRampMesh(
    Real halfExtent = 2.0, Real baseY = -1.5, Real slopeAngleDeg = 15.0) {
  auto mesh = std::make_shared<TriangleMesh>();

  Real slopeRad = slopeAngleDeg * glm::pi<Real>() / 180.0;
  Real rise = halfExtent * std::tan(slopeRad);

  // 四个顶点组成一个倾斜平面
  mesh->vertices = {
      {-halfExtent, baseY,         -halfExtent},  // 低端左
      { halfExtent, baseY + rise,  -halfExtent},  // 高端左
      { halfExtent, baseY + rise,   halfExtent},  // 高端右
      {-halfExtent, baseY,          halfExtent},  // 低端右
  };

  mesh->triangles = {
      {0, 1, 2},
      {0, 2, 3},
  };

  mesh->computeEdges();
  return mesh;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("mesh-collider",
      "Elastic cube falling onto a triangulated mesh collider");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("500"))
      ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
      ("ramp", "Use tilted ramp instead of flat ground", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const double dt = args["dt"].as<double>();
  const int maxSteps = args["steps"].as<int>();
  const bool noRender = args["no-render"].as<bool>();
  const bool useRamp = args["ramp"].as<bool>();

  // ─── 1. Mesh ────────────────────────────────────────────────────────────────
  auto meshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!meshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj" << std::endl;
    return 1;
  }
  auto tetMesh = std::move(*meshOpt);
  std::cout << std::format("Elastic body: {} vertices, {} tets\n",
                           tetMesh.getVertices().size(), tetMesh.tets.size());

  // ─── 2. Material ────────────────────────────────────────────────────────────
  auto energy = std::make_unique<StableNeoHookean<Real>>(
      ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});

  // ─── 3. System ──────────────────────────────────────────────────────────────
  System system;

  ElasticTetMesh elasticBody(std::move(tetMesh), std::move(energy), /*density=*/1000.0);
  system.addPrimitive(Primitive(std::move(elasticBody)));

  system.setGravity({0.0, -9.81, 0.0});

  // ─── 4. Mesh Collider ──────────────────────────────────────────────────────
  {
    Collider ground;

    std::shared_ptr<TriangleMesh> groundMesh;
    if (useRamp) {
      groundMesh = createRampMesh(3.0, -1.5, 15.0);
      std::cout << std::format("Collider: ramp mesh, {} vertices, {} triangles, {} edges\n",
                               groundMesh->vertices.size(),
                               groundMesh->triangles.size(),
                               groundMesh->edges.size());
    } else {
      groundMesh = createGroundMesh(3.0, -1.0, 4);
      std::cout << std::format("Collider: ground mesh, {} vertices, {} triangles, {} edges\n",
                               groundMesh->vertices.size(),
                               groundMesh->triangles.size(),
                               groundMesh->edges.size());
    }

    ground.geometry = Collider::MeshGeometry{groundMesh};
    ground.motion = staticMotion();
    ground.advanceTo(0.0);  // 初始化 currentVertices

    system.colliders().push_back(std::move(ground));
  }

  // 初始化系统（分配 DOF、构建质量矩阵等）
  system.init();

  // ─── 5. Integrator ──────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 0.01;  // 方块边长为2, dHat取边长的0.5%
  ipcConfig.contactStiffness = 1e6;  // kappa（配合较大dHat降低刚度）

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
      .windowTitle = useRamp ? "SimCraft - Mesh Collider (Ramp)"
                             : "SimCraft - Mesh Collider (Ground)",
  });

  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(system, step);
  };

  app.logInterval = 50;
  app.logFn = [&](int step) {
    Real T = system.kineticEnergy();
    Real V = system.potentialEnergy();
    std::cout << std::format("Step {:4d}, t = {:.4f}, T = {:.4e}, V = {:.4e}\n",
                             step, system.currentTime(), T, V);
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
