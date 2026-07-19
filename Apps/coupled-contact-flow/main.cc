#include <DER/der-scene.h>
#include <FEM/fem-scene.h>

#include <Runtime/global-solver.h>
#include <Renderer/simulation-app.h>

#include <cxxopts.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

ksk::engine::fem::TetMeshDesc makeTetObstacle()
{
  ksk::engine::fem::TetMeshDesc mesh;
  mesh.vertices = {
      glm::dvec3(-0.35, -0.25, 0.0),
      glm::dvec3(0.35, -0.25, 0.0),
      glm::dvec3(0.0, 0.35, 0.0),
      glm::dvec3(0.0, 0.0, 0.35),
  };
  mesh.tets = {std::array<int, 4>{0, 1, 2, 3}};
  mesh.surfaceEdges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{0, 2},
      std::array<int, 2>{0, 3},
      std::array<int, 2>{1, 2},
      std::array<int, 2>{1, 3},
      std::array<int, 2>{2, 3},
  };
  mesh.surfaceTriangles = {
      std::array<int, 3>{0, 1, 2},
      std::array<int, 3>{0, 1, 3},
      std::array<int, 3>{0, 2, 3},
      std::array<int, 3>{1, 2, 3},
  };
  return mesh;
}

ksk::der::DERRodDesc makeHairRod(const glm::dvec3& root,
                                 double length,
                                 double phase)
{
  ksk::der::DERRodDesc rod;
  constexpr int SEGMENTS = 32;
  rod.restBlocks.reserve(SEGMENTS + 1);
  for (int i = 0; i <= SEGMENTS; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(SEGMENTS);
    const double x = root.x + length * t;
    const double y = root.y + 0.035 * std::sin(3.14159265358979323846 * t);
    const double z =
        root.z + 0.035 * std::sin(6.28318530717958647692 * t + phase);
    rod.restBlocks.emplace_back(x, y, z, 0.0);
  }
  rod.material.density = 1300.0;
  rod.material.radius = 4.0e-5;
  rod.material.youngsModulus = 4.0e9;
  rod.material.shearModulus = 1.5e9;
  return rod;
}

ksk::core::Vec3f toVec3f(const glm::dvec3& v)
{
  return {
      static_cast<float>(v.x),
      static_cast<float>(v.y),
      static_cast<float>(v.z),
  };
}

void pinRodRoot(ksk::runtime::RuntimeSceneDesc& scene,
                ksk::runtime::ObjectRef rod,
                const glm::dvec3& root,
                double stiffness)
{
  scene.addConstraint(rod, "x", 0, stiffness, [x = root.x](double) {
    return x;
  });
  scene.addConstraint(rod, "y", 0, stiffness, [y = root.y](double) {
    return y;
  });
  scene.addConstraint(rod, "z", 0, stiffness, [z = root.z](double) {
    return z;
  });
}

std::unique_ptr<ksk::renderer::SceneProxy> buildProxy(
    const ksk::runtime::SimulationRunner& runner,
    int frame)
{
  const ksk::runtime::RuntimeScene& scene = runner.simulation().scene();
  const auto& geometry = scene.geometry;

  auto proxy = std::make_unique<ksk::renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = static_cast<float>(runner.time());
  proxy->camera.position = {0.20f, 0.25f, 1.25f};
  proxy->camera.target = {0.05f, 0.02f, 0.12f};

  std::vector<ksk::core::Vec3f> positions;
  positions.reserve(geometry.points.size());
  for (int point = 0; point < geometry.pointCount(); ++point) {
    positions.push_back(
        toVec3f(geometry.worldPosition(ksk::runtime::PointIdx{point})));
  }

  ksk::renderer::MeshProxy surface;
  surface.name = "soft-body-surface";
  surface.positions = positions;
  surface.objectColor = {0.28f, 0.55f, 0.92f};
  surface.triangles.reserve(geometry.triangles.size());
  for (const auto& triangle : geometry.triangles) {
    surface.triangles.push_back({
        static_cast<unsigned int>(triangle.p0),
        static_cast<unsigned int>(triangle.p1),
        static_cast<unsigned int>(triangle.p2),
    });
  }
  ksk::renderer::computeSmoothNormals(surface);
  proxy->meshes.push_back(std::move(surface));

  ksk::renderer::WireframeProxy edges;
  edges.name = "scene-edges";
  edges.color = {0.94f, 0.37f, 0.18f};
  edges.positions = positions;
  edges.edges.reserve(geometry.edges.size());
  for (const auto& edge : geometry.edges) {
    edges.edges.emplace_back(static_cast<unsigned int>(edge.p0),
                             static_cast<unsigned int>(edge.p1));
  }
  proxy->wireframes.push_back(std::move(edges));

  ksk::renderer::ParticleProxy points;
  points.name = "scene-points";
  points.positions = std::move(positions);
  points.radius = 0.0015f;
  points.color = {0.10f, 0.82f, 0.68f};
  proxy->particles.push_back(std::move(points));

  return proxy;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(
      "coupled-contact-flow",
      "Coupled DER/FEM contact scene with renderer");
  options.add_options()
      ("steps", "number of simulation steps",
       cxxopts::value<int>()->default_value("36000"))
      ("dt", "time step",
       cxxopts::value<double>()->default_value("0.005"))
      ("no-render", "disable rendering",
       cxxopts::value<bool>()->default_value("false"))
      ("h,help", "print help");

  const auto args = options.parse(argc, argv);
  if (args.count("help") > 0) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const int steps = args["steps"].as<int>();
  const double dt = args["dt"].as<double>();
  const bool no_render = args["no-render"].as<bool>();
  if (steps <= 0) {
    std::cerr << "coupled-contact-flow: steps must be positive\n";
    return 1;
  }
  if (dt <= 0.0) {
    std::cerr << "coupled-contact-flow: dt must be positive\n";
    return 1;
  }

  ksk::runtime::RuntimeSceneDesc scene;
  scene.gravity = glm::dvec3(0.0, -1.5, 0.0);
  scene.timeStep = dt;
  scene.solverConfig.maxNewtonIterations = 100;
  scene.solverConfig.maxPcgIterations = 1000;
  scene.solverConfig.maxLineSearchIterations = 8;
  scene.solverConfig.contactDetectionDistance = 1.0e-3;
  scene.solverConfig.contactBarrierDistance = 1.0e-3;
  scene.solverConfig.contactStiffness = 1.0e8;

  const ksk::runtime::ObjectRef tet =
      ksk::engine::fem::addTetMesh(scene, makeTetObstacle(), "tet-obstacle");
  ksk::engine::fem::addConstraint(
      scene, tet, "x", 0, 2e6, [](double) { return -0.35; });
  ksk::engine::fem::addConstraint(
      scene, tet, "y", 0, 2e6, [](double) { return -0.25; });
  ksk::engine::fem::addConstraint(
      scene, tet, "z", 0, 2e6, [](double) { return 0.0; });

  const ksk::runtime::ObjectRef hair0 =
      ksk::der::addRod(scene, makeHairRod({-0.55, 0.55, -0.16},
                                          0.95,
                                          0.0));
  const ksk::runtime::ObjectRef hair1 =
      ksk::der::addRod(scene, makeHairRod({-0.58, 0.62, 0.02},
                                          0.98,
                                          1.7));
  const ksk::runtime::ObjectRef hair2 =
      ksk::der::addRod(scene, makeHairRod({-0.52, 0.70, 0.18},
                                          0.90,
                                          3.2));
  pinRodRoot(scene, hair0, {-0.55, 0.55, -0.16}, 5000.0);
  pinRodRoot(scene, hair1, {-0.58, 0.62, 0.02}, 5000.0);
  pinRodRoot(scene, hair2, {-0.52, 0.70, 0.18}, 5000.0);

  ksk::runtime::SimulationRunner runner =
      ksk::runtime::buildSimulationRunner(scene);
  const ksk::runtime::RuntimeScene& initial_scene =
      runner.simulation().scene();

  std::cout << "coupled-contact-flow\n"
            << "  subsystems: " << initial_scene.subsystemBatches.size()
            << '\n'
            << "  dofs: " << initial_scene.dofs.totalScalars << '\n'
            << "  points: " << initial_scene.geometry.points.size() << '\n'
            << "  edges: " << initial_scene.geometry.edges.size() << '\n'
            << "  triangles: " << initial_scene.geometry.triangles.size()
            << '\n';

  bool solve_failed = false;
  auto step_once = [&](int step) {
    if (solve_failed) {
      return;
    }
    const ksk::runtime::RuntimeStepResult step_result = runner.step();
    const bool finite =
        std::isfinite(step_result.finalGradientNorm) &&
        std::isfinite(step_result.finalStepNorm);
    if (!finite)
    {
        solve_failed = true;
    }
      std::cout << "step=" << step + 1
                << " iterations=" << step_result.iterations
                << " converged="
                << (step_result.converged ? "true" : "false")
                << " grad=" << step_result.finalGradientNorm
                << " step_norm=" << step_result.finalStepNorm
                << '\n';
  };

  if (no_render) {
    for (int step = 0; step < steps && !solve_failed; ++step) {
      step_once(step);
    }
    return solve_failed ? 2 : 0;
  }

  ksk::renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "Kiseki - Coupled DER/FEM Contact",
  });
  app.stepFn = [&](int step) {
    step_once(step);
  };
  app.buildProxy = [&](int frame) {
    return buildProxy(runner, frame);
  };
  app.logInterval = 30;
  app.run(steps);

  return solve_failed ? 2 : 0;
}
