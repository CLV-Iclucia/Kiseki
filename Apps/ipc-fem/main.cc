//
// Created by creeper on 5/23/24.
//
#include <cxxopts.hpp>
#include <fem/fem-simulation.h>
#include <Renderer/simulation-app.h>
#include <fem/scene-proxy.h>
#include <iostream>
#include <format>

using namespace ksk;
using namespace ksk::fem;

void checkArgs(const cxxopts::ParseResult& result) {
  if (!result.count("input")) {
    std::cerr << "Please specify input file" << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  cxxopts::Options options("IPC FEM", "FEM Soft body simulator using IPC");
  options.add_options()
      ("i,input", "Input file", cxxopts::value<std::string>())
      ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print help");
  auto result = options.parse(argc, argv);
  checkArgs(result);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  auto inputFile = result["input"].as<std::string>();
  auto simBuilder = FEMSimulationBuilder{};
  auto simConfig = core::loadJsonFile(inputFile);

  if (!simConfig) {
    std::cerr << "Failed to load simulation configuration from " << inputFile
              << std::endl;
    return 1;
  }

  auto femSim = simBuilder.build(*simConfig);
  core::Frame frame;

  bool useRenderer = !result["no-render"].as<bool>();

  // ─── Run ────────────────────────────────────────────────────────────────────

  renderer::SimulationApp app({.windowTitle = "SimCraft - IPC FEM"});

  app.stepFn = [&](int) {
    femSim.step(frame);
  };

  app.buildProxy = [&](int step) {
    return renderer::buildSceneProxyFromSystem(femSim.getSystem(), step);
  };

  app.logInterval = 10;
  app.logFn = [&](int step) {
    std::cout << std::format("Simulated frame {}, time = {:.4f}\n",
                             step, femSim.getSystem().currentTime());
  };

  int completed;
  if (!useRenderer) {
    completed = app.runHeadless(1000);
  } else {
    completed = app.run(1000);
  }

  std::cout << std::format("Application terminated. {} steps completed.\n", completed);
  return 0;
}
