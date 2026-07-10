//
// Created by CreeperIclucia-Vader on 25-5-29.
//
#include <Maths/linear-solver-factory.h>
#include <spdlog/spdlog.h>

namespace ksk::maths {

LinearSolverFactory& LinearSolverFactory::instance() {
  static LinearSolverFactory instance;
  return instance;
}

void LinearSolverFactory::registerCreator(const std::string &name,
                                          Creator creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (creators_.find(name) != creators_.end())
    throw std::runtime_error("Duplicate registration: " + name);
  creators_[name] = std::move(creator);
  spdlog::info("Registered LinearSolver: {}", name);
}

std::unique_ptr<LinearSolver>
LinearSolverFactory::create(const std::string &name,
                            const core::JsonNode &json) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (auto it = creators_.find(name); it != creators_.end()) {
    try {
      return it->second(json);
    } catch (const std::exception &e) {
      throw std::runtime_error("Creation failed for '" + name +
                               "': " + e.what());
    }
  }
  throw std::runtime_error("Unknown type: " + name);
}

} 