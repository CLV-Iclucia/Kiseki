//
// Created by CreeperIclucia-Vader on 25-5-29.
//

#pragma once
#include <Maths/linear-solver.h>
#include <mutex>

namespace ksk::maths {

struct LinearSolverFactory {
  using Creator = std::function<std::unique_ptr<LinearSolver>(const core::JsonNode&)>;

  LinearSolverFactory(const LinearSolverFactory&) = delete;
  void operator=(const LinearSolverFactory&) = delete;
  static LinearSolverFactory& instance();
  void registerCreator(const std::string& name, Creator creator);
  std::unique_ptr<LinearSolver>
  create(const std::string &name, const core::JsonNode &json) const;

private:
  LinearSolverFactory() = default;
  std::unordered_map<std::string, Creator> creators_;
  mutable std::mutex mutex_;
};

} 