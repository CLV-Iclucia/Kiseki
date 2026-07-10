//
// SimCraft Example: FEM Showcase (C++)
// =====================================
//
// A comprehensive FEM test scene demonstrating multiple features:
//   - Multiple elastic bodies (1 cube + 2 bunnies)
//   - Mesh collider ground plane (triangulated)
//   - Body-body IPC collisions (cube smashes bunny, bunny rams bunny)
//   - Mixed geometry complexity (cube vs high-res bunny mesh)
//
// Future extensions (marked with TODO):
//   - Per-body material variations (different E, nu, constitutive models)
//   - Boundary constraints (pinVertices, prescribeMotion)
//
// Usage:
//   fem-showcase [--dt 0.01] [--steps 1000] [--no-render]
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

// ─── Helper: 将网格平移 ──────────────────────────────────────────────────────
static TetMesh translateMesh(const TetMesh& src, Vector<Real, 3> offset) {
  auto verts = src.getVertices();
  for (auto& v : verts) {
    v += offset;
  }
  return TetMesh(verts, src.tets);
}

// ─── Helper: 将网格平移并赋初速度 ────────────────────────────────────────────
static TetMesh translateMeshWithVelocity(const TetMesh& src, Vector<Real, 3> offset,
                                         Vector<Real, 3> velocity) {
  auto verts = src.getVertices();
  for (auto& v : verts) {
    v += offset;
  }
  std::vector<Vector<Real, 3>> vels(verts.size(), velocity);
  return TetMesh(verts, src.tets, std::move(vels));
}

// ─── Helper: 创建地面三角网格 ────────────────────────────────────────────────
static std::shared_ptr<TriangleMesh> createGroundMesh(
    Real halfExtent = 5.0, Real groundY = -2.0, int gridN = 6) {
  auto mesh = std::make_shared<TriangleMesh>();

  int nSide = gridN + 1;
  mesh->vertices.reserve(nSide * nSide);
  for (int j = 0; j < nSide; ++j) {
    for (int i = 0; i < nSide; ++i) {
      Real u = static_cast<Real>(i) / gridN;
      Real v = static_cast<Real>(j) / gridN;
      Real x = (u - 0.5) * 2.0 * halfExtent;
      Real z = (v - 0.5) * 2.0 * halfExtent;
      mesh->vertices.emplace_back(x, groundY, z);
    }
  }

  mesh->triangles.reserve(gridN * gridN * 2);
  for (int j = 0; j < gridN; ++j) {
    for (int i = 0; i < gridN; ++i) {
      int v00 = j * nSide + i;
      int v10 = j * nSide + (i + 1);
      int v01 = (j + 1) * nSide + i;
      int v11 = (j + 1) * nSide + (i + 1);
      mesh->triangles.emplace_back(glm::ivec3(v00, v10, v11));
      mesh->triangles.emplace_back(glm::ivec3(v00, v11, v01));
    }
  }

  return mesh;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  cxxopts::Options options("fem-showcase",
      "Comprehensive FEM test: multi-body collisions on mesh ground (cube + 2 bunnies)");
  options.add_options()
      ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
      ("steps", "Number of simulation steps", cxxopts::value<int>()->default_value("1000"))
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

  // ─── 1. Load meshes ─────────────────────────────────────────────────────────
  auto cubeMeshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  if (!cubeMeshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/cube10x10.tobj\n";
    return 1;
  }
  auto cubeMesh = std::move(*cubeMeshOpt);
  std::cout << std::format("Cube mesh: {} vertices, {} tets\n",
                           cubeMesh.getVertices().size(),
                           cubeMesh.tets.size());

  auto bunnyMeshOpt = readTetMeshFromTOBJ(FEM_TETS_DIR "/bunny.tobj");
  if (!bunnyMeshOpt) {
    std::cerr << "Failed to load mesh: " FEM_TETS_DIR "/bunny.tobj\n";
    return 1;
  }
  auto bunnyMesh = std::move(*bunnyMeshOpt);
  std::cout << std::format("Bunny mesh: {} vertices, {} tets\n",
                           bunnyMesh.getVertices().size(),
                           bunnyMesh.tets.size());

  // ─── 2. Create elastic bodies ──────────────────────────────────────────────
  //
  // TODO [材料扩展]: 目前所有物体使用相同的 StableNeoHookean 材料。
  //   未来可以为每个物体指定不同的本构模型和参数，例如：
  //     - 软体: StableNeoHookean, E=1e4, nu=0.45
  //     - 硬体: StableNeoHookean, E=1e6, nu=0.3
  //     - ARAP 或 LinearElastic 模型
  //   只需为每个 ElasticTetMesh 传入不同的 energy 即可。
  //

  System system;

  // Body A: 高处立方体，自由落体，砸向下方的 bunny
  {
    auto mesh = translateMesh(cubeMesh, Vector<Real, 3>(0.0, 4.0, 0.0));
    auto energy = std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});
    ElasticTetMesh body(std::move(mesh), std::move(energy), /*density=*/1000.0);
    system.addPrimitive(Primitive(std::move(body)));
    std::cout << "  Body A: cube at y=4.0, free-fall\n";
  }

  // Body B: bunny，坐在地面上方，等待被立方体砸中
  {
    auto mesh = translateMesh(bunnyMesh, Vector<Real, 3>(0.0, 1.0, 0.0));
    auto energy = std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});
    ElasticTetMesh body(std::move(mesh), std::move(energy), /*density=*/1000.0);
    system.addPrimitive(Primitive(std::move(body)));
    std::cout << "  Body B: bunny at y=1.0, resting above ground\n";
  }

  // Body C: 第二只 bunny，从侧面飞入与 Body B 发生碰撞
  {
    auto mesh = translateMeshWithVelocity(bunnyMesh,
                                          Vector<Real, 3>(-4.0, 1.0, 0.0),   // 位置: 左侧远处
                                          Vector<Real, 3>(3.0, 0.0, 0.0));   // 速度: 向右飞
    auto energy = std::make_unique<StableNeoHookean<Real>>(
        ElasticityParameters<Real>{.E = 1e5, .nu = 0.4});
    ElasticTetMesh body(std::move(mesh), std::move(energy), /*density=*/800.0);
    system.addPrimitive(Primitive(std::move(body)));
    std::cout << "  Body C: bunny at x=-4, flying rightward\n";
  }

  system.setGravity(glm::dvec3(0.0, -9.81, 0.0));

  // ─── 3. Colliders ──────────────────────────────────────────────────────────

  // Ground — 三角网格碰撞体 (MeshGeometry)
  {
    Collider ground;
    auto groundMesh = createGroundMesh(/*halfExtent=*/5.0, /*groundY=*/-1.0, /*gridN=*/6);
    ground.geometry = Collider::MeshGeometry{groundMesh};
    ground.motion = staticMotion();
    ground.advanceTo(0.0);

    system.colliders().push_back(std::move(ground));
    std::cout << std::format("  Collider: ground mesh ({} verts, {} tris)\n",
                             groundMesh->vertices.size(),
                             groundMesh->triangles.size());
  }

  // ─── 4. Constraints (预留) ─────────────────────────────────────────────────
  //
  // TODO [约束扩展]: 目前无边界约束。未来可以添加：
  //   - system.constraints().pinVertices(indices, system.x)
  //       固定某些顶点（如将一个物体的底面钉住做悬臂梁）
  //   - system.constraints().prescribeMotion(vertexIdx, posFunc, velFunc)
  //       指定某些顶点做正弦/线性运动
  //   - system.constraints().pinComponent(vertexIdx, 1, value)
  //       约束单个方向分量
  //   添加约束后需调用: system.constraints().build(system.x.numBlocks())
  //

  // ─── 5. Initialize system ──────────────────────────────────────────────────
  system.init();
  std::cout << std::format("\nSystem initialized: {} total DOF\n", system.dof());

  // ─── 6. Integrator ─────────────────────────────────────────────────────────
  IpcIntegrator::Config ipcConfig;
  ipcConfig.dHat = 1e-2;
  ipcConfig.contactStiffness = 1e8;  // 多体场景需要较高 kappa

  auto integrator = std::make_unique<IpcImplicitEuler>(system, ipcConfig);
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(
      /*maxIter=*/2000, /*tol=*/1e-6);

  std::cout << std::format("IPC integrator: dHat = {}, kappa = {}\n",
                           ipcConfig.dHat, ipcConfig.contactStiffness);
  std::cout << std::format("Simulation: dt = {}, max_steps = {}\n\n", dt, maxSteps);

  // ─── 7. Run ────────────────────────────────────────────────────────────────

  renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - FEM Showcase",
  });

  app.stepFn = [&](int) {
    integrator->step(dt);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(system, step);
  };

  app.logInterval = 25;
  app.logFn = [&](int step) {
    Real KE = system.kineticEnergy();
    Real PE = system.potentialEnergy();
    Real total = KE + PE;
    std::cout << std::format(
        "Step {:4d} | t = {:.3f}s | KE = {:.4e} | PE = {:.4e} | Total = {:.4e}\n",
        step, system.currentTime(), KE, PE, total);
  };

  int completed;
  if (noRender) {
    completed = app.runHeadless(maxSteps);
  } else {
    completed = app.run(maxSteps);
  }

  std::cout << std::format("\nSimulation finished. {} / {} steps completed.\n",
                           completed, maxSteps);
  return 0;
}
