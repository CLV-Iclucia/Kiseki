//
// Created by CreeperIclucia-Vader on 25-5-26.
//
#include <Maths/linear-solvers/cg-solver.h>
#include <Maths/linear-solver-factory.h>

namespace ksk::maths {

int cg_auto_reg = ([]() {
    LinearSolverFactory::instance().registerCreator(
        "cg-solver",
        [](const core::JsonNode &json) {
          return ConjugateGradientSolver::createFromJson(json);
        });
  }(), 0);

std::unique_ptr<LinearSolver>
ConjugateGradientSolver::createFromJson(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("ConjugateGradientSolver requires a JSON object");
  const auto &dict = json.as<core::JsonDict>();
  auto solver = std::make_unique<ConjugateGradientSolver>();
  if (dict.contains("preconditioner")) {
    auto preconditionerJson = dict.at("preconditioner");
    solver->preconditioner = createLinearSolver(preconditionerJson);
  } else
    solver->preconditioner = nullptr;
  if (dict.contains("maxIterations")) {
    int maxIterations = dict.at("maxIterations").as<int>();
    if (maxIterations <= 0)
      throw std::runtime_error("maxIterations must be positive");
    solver->solver.setMaxIterations(maxIterations);
  }
  if (dict.contains("tolerance")) {
    Real tolerance = dict.at("tolerance").as<Real>();
    if (tolerance <= 0.0)
      throw std::runtime_error("tolerance must be positive");
    solver->solver.setTolerance(tolerance);
  }
  return solver;
}

} // namespace maths