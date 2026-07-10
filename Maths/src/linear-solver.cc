//
// Created by CreeperIclucia-Vader on 25-5-28.
//
#include <Maths/linear-solver-factory.h>

namespace ksk::maths {

std::unique_ptr<LinearSolver>
createLinearSolver(const core::JsonNode &node) {
  if (!node.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary for LinearSolver");

  const auto &dict = node.as<core::JsonDict>();
  const auto &type = dict.at("type").as<std::string>();
  return LinearSolverFactory::instance().create(type, node);
}

} // namespace ksk::maths 