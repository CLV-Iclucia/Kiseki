#include <Runtime/global-solver.h>
#include <Renderer/simulation-app.h>
#include <SceneObjects/hair.h>

#include <cxxopts.hpp>
#include <glm/glm.hpp>

#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

ksk::scene::HairObjectDesc makeHair(int segments,
                                    double length,
                                    double yOffset,
                                    double zOffset,
                                    const ksk::scene::HairMaterial& material)
{
  if (segments < 2) {
    throw std::invalid_argument("hair strand requires at least two segments");
  }

  ksk::scene::HairObjectDesc hair("hair");
  hair.material = material;
  hair.restBlocks.reserve(static_cast<size_t>(segments + 1));
  const double amplitude = length / 15.0;
  for (int vertex = 0; vertex <= segments; ++vertex) {
    const double t = static_cast<double>(vertex) / static_cast<double>(segments);
    const double x = length * t;
    const double y =
        yOffset + amplitude * std::sin(std::numbers::pi * t);
    const double z =
        zOffset + 0.25 * amplitude *
                      std::sin(2.0 * std::numbers::pi * t);
    hair.restBlocks.emplace_back(x, y, z, 0.0);
  }
  return hair;
}

void printTipPositions(const ksk::runtime::SimulationRunner& simulation,
                       const std::vector<int>& hairVertexCounts)
{
  const auto& geometry = simulation.simulation().scene().geometry;
  int offset = 0;
  for (int hair = 0; hair < static_cast<int>(hairVertexCounts.size()); ++hair) {
    offset += hairVertexCounts[static_cast<size_t>(hair)];
    const glm::dvec3 tip =
        geometry.worldPosition(ksk::runtime::GeometryPointId{offset - 1});
    std::cout << " hair" << hair << "_tip=("
              << tip.x << ", " << tip.y << ", " << tip.z << ")";
  }
}

ksk::core::Vec3f toVec3f(const glm::dvec3& v)
{
  return {
      static_cast<float>(v.x),
      static_cast<float>(v.y),
      static_cast<float>(v.z),
  };
}

void addTwistTestConstraints(ksk::runtime::RuntimeSceneDesc& scene,
                             ksk::runtime::ObjectRef hairRef,
                             double rootStiffness,
                             double twistStiffness,
                             const double& twistTarget)
{
  const auto* hair =
      scene.findObjectDescAs<ksk::scene::HairObjectDesc>(hairRef);
  if (!hair) {
    throw std::runtime_error("cannot pin root of a non-hair object");
  }
  if (hair->restBlocks.empty()) {
    throw std::runtime_error("cannot pin root of an empty hair object");
  }

  const ksk::scene::HairBlock& root_block = hair->restBlocks.front();
  const int terminal_twist_sample =
      static_cast<int>(hair->restBlocks.size()) - 2;
  for (const ksk::runtime::ObjectRef rod : hair->rodRefs(scene)) {
    scene.addConstraint(rod, "x", 0, rootStiffness,
                        [x = root_block.x](double) { return x; });
    scene.addConstraint(rod, "y", 0, rootStiffness,
                        [y = root_block.y](double) { return y; });
    scene.addConstraint(rod, "z", 0, rootStiffness,
                        [z = root_block.z](double) { return z; });
    scene.addConstraint(rod, "twist", 0, twistStiffness,
                        [theta = root_block.w](double) { return theta; });
    scene.addConstraint(rod, "twist", terminal_twist_sample, twistStiffness,
                        [&twistTarget](double) { return twistTarget; });
  }
}

std::unique_ptr<ksk::renderer::SceneProxy> buildProxy(
    const ksk::runtime::SimulationRunner& simulation,
    int frame,
    double dt)
{
  auto proxy = std::make_unique<ksk::renderer::SceneProxy>();
  proxy->frameIndex = frame;
  proxy->simulationTime = static_cast<float>(frame * dt);
  proxy->camera.position = {0.08f, 0.08f, 0.20f};
  proxy->camera.target = {0.02f, 0.0f, 0.0f};

  const auto& geometry = simulation.simulation().scene().geometry;

  ksk::renderer::WireframeProxy hairs;
  hairs.name = "hairs";
  hairs.color = {0.86f, 0.33f, 0.22f};
  hairs.positions.reserve(geometry.points.size());
  for (int point = 0; point < geometry.pointCount(); ++point) {
    hairs.positions.push_back(
        toVec3f(geometry.worldPosition(ksk::runtime::GeometryPointId{point})));
  }
  hairs.edges.reserve(geometry.edges.size());
  for (const auto& edge : geometry.edges) {
    hairs.edges.emplace_back(static_cast<unsigned int>(edge.p0),
                             static_cast<unsigned int>(edge.p1));
  }
  proxy->wireframes.push_back(std::move(hairs));

  ksk::renderer::ParticleProxy vertices;
  vertices.name = "hair-vertices";
  vertices.radius = 0.0001f;
  vertices.color = hairs.color;
  vertices.positions.reserve(geometry.points.size());
  for (int point = 0; point < geometry.pointCount(); ++point) {
    vertices.positions.push_back(
        toVec3f(geometry.worldPosition(ksk::runtime::GeometryPointId{point})));
  }
  proxy->particles.push_back(std::move(vertices));

  return proxy;
}

}  // namespace

int main(int argc, char** argv)
{
  cxxopts::Options options(
      "der-hair",
      "Runtime simulation with hair strands");
  options.add_options()
      ("steps", "number of simulation steps",
       cxxopts::value<int>()->default_value("12000"))
      ("dt", "time step",
       cxxopts::value<double>()->default_value("0.001"))
      ("hairs", "number of hair rods",
       cxxopts::value<int>()->default_value("1"))
      ("segments", "segments per rod",
       cxxopts::value<int>()->default_value("127"))
      ("length", "rod length",
       cxxopts::value<double>()->default_value("1.2"))
      ("root-stiffness", "root position constraint stiffness",
       cxxopts::value<double>()->default_value("200000"))
      ("twist-stiffness", "twist constraint stiffness",
       cxxopts::value<double>()->default_value("2000000"))
      ("twist-rate", "terminal twist target rate in radians per second",
       cxxopts::value<double>()->default_value("6.283185307179586"))
      
      ("no-render", "disable rendering",
       cxxopts::value<bool>()->default_value("false"))
      ("gravity", "enable gravity")
      ("help", "print help");

  const auto args = options.parse(argc, argv);
  if (args.count("help") > 0) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const int steps = args["steps"].as<int>();
  const double time_step = args["dt"].as<double>();
  const int hair_count = args["hairs"].as<int>();
  const int segments = args["segments"].as<int>();
  const double length = args["length"].as<double>();
  const double root_stiffness = args["root-stiffness"].as<double>();
  const double twist_stiffness = args["twist-stiffness"].as<double>();
  const double twist_rate = args["twist-rate"].as<double>();
  const bool no_render = args["no-render"].as<bool>();

  if (steps <= 0) {
    throw std::invalid_argument("steps must be positive");
  }
  if (time_step <= 0.0) {
    throw std::invalid_argument("dt must be positive");
  }
  if (hair_count <= 0) {
    throw std::invalid_argument("hairs must be positive");
  }
  if (twist_stiffness <= 0.0) {
    throw std::invalid_argument("twist-stiffness must be positive");
  }

  ksk::runtime::RuntimeSceneDesc scene;
  scene.timeStep = time_step;
  scene.gravity =
      args.count("gravity") > 0 ? glm::dvec3(0.0, -9.81, 0.0)
                                : glm::dvec3(0.0);
  scene.solverConfig.maxNewtonIterations = 20;
  scene.solverConfig.maxPcgIterations = 100;
  scene.solverConfig.newtonGradientTolerance = 1.0e-8;
  scene.solverConfig.newtonStepTolerance = 1.0e-10;
  double twist_target = 0.0;

  ksk::scene::HairMaterial material;
  material.radius = 4.0e-5;
  material.density = 1300.0;
  material.youngsModulus = 4.0e9;
  material.shearModulus = 1.5e9;

  std::vector<int> hair_vertex_counts;
  hair_vertex_counts.reserve(static_cast<size_t>(hair_count));
  const double spacing = 0.12 * length;
  const double center = 0.5 * static_cast<double>(hair_count - 1);
  for (int hair = 0; hair < hair_count; ++hair) {
    const double u = static_cast<double>(hair) - center;
    const double y = u * spacing;
    const double phase =
        6.283185307179586 * static_cast<double>(hair) /
        static_cast<double>(hair_count);
    const double z = 0.05 * length * std::sin(phase);
    auto strand = makeHair(segments, length, y, z, material);
    hair_vertex_counts.push_back(static_cast<int>(strand.restBlocks.size()));
    const ksk::runtime::ObjectRef hairRef =
        ksk::scene::addHair(scene, std::move(strand));
    addTwistTestConstraints(scene, hairRef, root_stiffness,
                            twist_stiffness, twist_target);
  }

  ksk::runtime::SimulationRunner simulation =
      ksk::runtime::buildSimulationRunner(scene);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "hair simulation: hairs=" << hair_count
            << " segments=" << segments
            << " root_stiffness=" << root_stiffness
            << " twist_stiffness=" << twist_stiffness
            << " twist_rate=" << twist_rate
            << " dofs=" << simulation.simulation().scene().dofs.totalScalars
            << " geometry_points="
            << simulation.simulation().scene().geometry.points.size()
            << '\n';

  bool solve_failed = false;
  auto step_once = [&](int step) -> bool {
    if (solve_failed) {
      return false;
    }

    twist_target = twist_rate * static_cast<double>(step + 1) * time_step;
    const ksk::runtime::RuntimeStepResult result = simulation.step();

    std::cout << "step=" << (step + 1)
              << " iterations=" << result.iterations
              << " converged=" << (result.converged ? "true" : "false")
              << " grad=" << std::scientific << std::setprecision(6)
              << result.finalGradientNorm
              << " step_norm=" << result.finalStepNorm
              << std::fixed << std::setprecision(6);
    printTipPositions(simulation, hair_vertex_counts);
    std::cout << '\n';

    if (!result.converged) {
      std::cerr << std::format(
          "hair solve did not converge at step {}\n", step + 1);
      solve_failed = true;
      return false;
    }
    return true;
  };

  if (no_render) {
    for (int step = 0; step < steps; ++step) {
      if (!step_once(step)) {
        return 2;
      }
    }
    return 0;
  }

  ksk::renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "Kiseki - Discrete Elastic Rods",
  });
  app.stepFn = [&](int step) {
    step_once(step);
  };
  app.buildProxy = [&](int frame) {
    return buildProxy(simulation, frame, time_step);
  };
  app.logInterval = 20;
  app.run(steps);

  return solve_failed ? 2 : 0;
}
