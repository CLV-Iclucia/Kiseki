#include <HairSim/hair-sim.h>
#include <Renderer/simulation-app.h>

#include <cxxopts.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <iterator>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace {

using sim::hairsim::ImplicitRodIntegrator;
using sim::hairsim::Rod;
using sim::hairsim::RodBlock;
using sim::hairsim::RodMaterial;

std::vector<RodBlock> straightRod(int vertices, const glm::dvec3& origin,
                                  double length) {
  std::vector<RodBlock> blocks;
  blocks.reserve(static_cast<size_t>(vertices));
  for (int i = 0; i < vertices; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(vertices - 1);
    blocks.emplace_back(origin.x + length * u, origin.y, origin.z, 0.0);
  }
  return blocks;
}

std::vector<RodBlock> helicalRod(int vertices, const glm::dvec3& origin,
                                 double radius, double pitch,
                                 double turns) {
  std::vector<RodBlock> blocks;
  blocks.reserve(static_cast<size_t>(vertices));
  const double angle_end = 2.0 * std::numbers::pi * turns;
  for (int i = 0; i < vertices; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(vertices - 1);
    const double angle = angle_end * u;
    blocks.emplace_back(
        origin.x + pitch * turns * u,
        origin.y + radius * std::cos(angle),
        origin.z + radius * std::sin(angle),
        0.0);
  }
  return blocks;
}

std::unique_ptr<Rod> makeRod(const std::string& scenario, int vertices,
                             const glm::dvec3& origin, int index) {
  RodMaterial material;
  material.rootStiffness = 2e3;

  std::vector<RodBlock> rest_blocks =
      scenario == "helix"
          ? helicalRod(vertices, origin, 0.012, 0.012, 3.0)
          : straightRod(vertices, origin, 1.2);
  auto rod = std::make_unique<Rod>(std::move(rest_blocks), material);
  if (scenario == "stretch") {
    for (int i = 1; i < vertices; ++i) {
      auto position = rod->state().position(static_cast<size_t>(i));
      position.x = origin.x + 1.45 * (position.x - origin.x);
      rod->state().setPosition(static_cast<size_t>(i), position);
    }
  } else if (scenario == "bend") {
    for (int i = 1; i < vertices; ++i) {
      const double u = static_cast<double>(i) / (vertices - 1);
      auto position = rod->state().position(static_cast<size_t>(i));
      position.y += 0.3 * u * u;
      rod->state().setPosition(static_cast<size_t>(i), position);
    }
  } else if (scenario == "twist") {
    for (int i = 0; i + 1 < vertices; ++i) {
      const double u = static_cast<double>(i) / (vertices - 1);
      rod->state().setTheta(static_cast<size_t>(i), 2000.0 * u);
    }
  }
  rod->resetReferenceFrames();
  return rod;
}

glm::dvec3 initialDirector(const glm::dvec3& tangent) {
  const glm::dvec3 axis =
      std::abs(tangent.z) < 0.9 ? glm::dvec3(0.0, 0.0, 1.0)
                                : glm::dvec3(0.0, 1.0, 0.0);
  return glm::normalize(glm::cross(axis, tangent));
}

glm::dvec3 parallelTransport(const glm::dvec3& director,
                             const glm::dvec3& from,
                             const glm::dvec3& to) {
  const glm::dvec3 axis = glm::cross(from, to);
  const double denominator = 1.0 + glm::dot(from, to);
  if (denominator < 1e-7) return initialDirector(to);
  return glm::normalize(
      director + glm::cross(axis, director) +
      glm::cross(axis, glm::cross(axis, director)) / denominator);
}

std::unique_ptr<sim::renderer::SceneProxy> buildProxy(
    const std::vector<std::unique_ptr<Rod>>& rods, int frame, double dt,
    bool show_frames) {
  auto scene = std::make_unique<sim::renderer::SceneProxy>();
  scene->frameIndex = frame;
  scene->simulationTime = static_cast<float>(frame * dt);
  scene->camera.position = {0.08f, 0.08f, 0.20f};
  scene->camera.target = {0.02f, 0.0f, 0.0f};

  static constexpr glm::vec3 COLORS[] = {
      {0.88f, 0.28f, 0.22f},
      {0.15f, 0.55f, 0.78f},
      {0.20f, 0.68f, 0.38f},
      {0.83f, 0.58f, 0.16f},
  };
  for (size_t r = 0; r < rods.size(); ++r) {
    sim::renderer::WireframeProxy wire;
    wire.name = std::format("rod-{}", r);
    wire.color = COLORS[r % std::size(COLORS)];
    const auto& state = rods[r]->state();
    wire.positions.reserve(state.size());
    for (size_t i = 0; i < state.size(); ++i)
      wire.positions.emplace_back(state.position(i));
    for (uint32_t i = 0; i + 1 < state.size(); ++i)
      wire.edges.emplace_back(i, i + 1);
    scene->wireframes.push_back(std::move(wire));

    sim::renderer::ParticleProxy particles;
    particles.name = std::format("vertices-{}", r);
    particles.positions = scene->wireframes.back().positions;
    particles.radius = 0.0015f;
    particles.color = COLORS[r % std::size(COLORS)];
    scene->particles.push_back(std::move(particles));

    if (show_frames) {
      sim::renderer::WireframeProxy frames;
      frames.name = std::format("frames-{}", r);
      frames.color = {0.25f, 0.85f, 0.75f};
      std::vector<glm::dvec3> tangents(state.size() - 1);
      std::vector<glm::dvec3> references(state.size() - 1);
      for (size_t i = 0; i + 1 < state.size(); ++i)
        tangents[i] =
            glm::normalize(state.position(i + 1) - state.position(i));
      references[0] = initialDirector(tangents[0]);
      for (size_t i = 1; i < references.size(); ++i)
        references[i] =
            parallelTransport(references[i - 1], tangents[i - 1], tangents[i]);

      for (size_t i = 0; i + 1 < state.size(); ++i) {
        const glm::dvec3 p0 = state.position(i);
        const glm::dvec3 p1 = state.position(i + 1);
        const glm::dvec3 second = glm::cross(tangents[i], references[i]);
        const double theta = state.theta(i);
        const glm::dvec3 director =
            std::cos(theta) * references[i] + std::sin(theta) * second;
        const glm::dvec3 center = 0.5 * (p0 + p1);
        const uint32_t base = static_cast<uint32_t>(frames.positions.size());
        frames.positions.emplace_back(center);
        frames.positions.emplace_back(center + 0.06 * director);
        frames.edges.emplace_back(base, base + 1);
      }
      scene->wireframes.push_back(std::move(frames));
    }
  }
  return scene;
}

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options("hair-sim", "CPU discrete elastic rod validation app");
  options.add_options()
      ("scenario", "hanging, stretch, bend, twist, helix, or multi",
       cxxopts::value<std::string>()->default_value("helix"))
      ("vertices", "Vertices per rod",
       cxxopts::value<int>()->default_value("32"))
      ("dt", "Time step", cxxopts::value<double>()->default_value("0.001"))
      ("steps", "Number of steps", cxxopts::value<int>()->default_value("10000"))
      ("no-render", "Disable rendering",
       cxxopts::value<bool>()->default_value("false"))
      ("show-frames", "Show material-frame directors",
       cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  if (args.count("help")) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const std::string scenario = args["scenario"].as<std::string>();
  const int vertices = std::max(3, args["vertices"].as<int>());
  const double dt = args["dt"].as<double>();
  const int steps = std::max(0, args["steps"].as<int>());
  const bool no_render = args["no-render"].as<bool>();
  const bool show_frames = args["show-frames"].as<bool>();

  const int rod_count = scenario == "multi" ? 4 : 1;
  std::vector<std::unique_ptr<Rod>> rods;
  std::vector<std::unique_ptr<ImplicitRodIntegrator>> integrators;
  for (int i = 0; i < rod_count; ++i) {
    const glm::dvec3 origin(0.0, 0.035 * i, 0.0);
    rods.push_back(makeRod(scenario, vertices, origin, i));
    integrators.push_back(
        std::make_unique<ImplicitRodIntegrator>(*rods.back()));
  }

  sim::renderer::SimulationApp app({
      .windowWidth = 1280,
      .windowHeight = 720,
      .windowTitle = "SimCraft - Discrete Elastic Rods",
  });
  app.stepFn = [&](int step) {
    for (size_t i = 0; i < rods.size(); ++i) {
      if (!integrators[i]->step(dt, {0.0, -9.81, 0.0})) {
        std::cerr << std::format(
            "[HairSim] step {} rod {} failed: {}\n", step, i,
            integrators[i]->diagnostic());
      }
    }
  };
  app.buildProxy = [&](int frame) {
    return buildProxy(rods, frame, dt, show_frames);
  };
  app.logInterval = 20;
  app.logFn = [&](int step) {
    const auto& energy = integrators.front()->lastEvaluation().energy;
    std::cout << std::format(
        "[HairSim] step {:5d} E={:.8g} stretch={:.4g} bend={:.4g} "
        "twist={:.4g} gravity={:.4g} pin={:.4g}\n",
        step, energy.total(), energy.stretching, energy.bending,
        energy.twisting, energy.gravity, energy.constraint);
  };

  if (no_render)
    app.runHeadless(steps);
  else
    app.run(steps);
  return 0;
}
