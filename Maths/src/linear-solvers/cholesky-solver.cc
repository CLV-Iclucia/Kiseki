//
// Created by CreeperIcludia-Vader on 25-5-26.
//
#include <Maths/linear-solvers/cholesky-solver.h>
#include <Maths/linear-solver-factory.h>

namespace ksk::maths {

int cholesky_auto_reg = ([]() {
    LinearSolverFactory::instance().registerCreator(
        "cholesky-solver",
        [](const core::JsonNode &json) {
          return SparseCholeskySolver::createFromJson(json);
        });
  }(), 0);

std::unique_ptr<LinearSolver>
SparseCholeskySolver::createFromJson(const core::JsonNode &json) {
  auto solver = std::make_unique<SparseCholeskySolver>();
  return solver;
}

}