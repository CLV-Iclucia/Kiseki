//
// Created by CreeperIclucia-Vader on 25-5-16.
//

#pragma once

#include <Maths/linear-solver.h>

namespace ksk::maths {

struct ConjugateGradientSolver final : LinearSolver {
  Vector<Real, Dynamic> solve(const SparseMatrix<Real> &A,
                              const Vector<Real, Dynamic> &b) override {
    if (preconditioner) {
      solver.compute(A);
    } else {
      solver.compute(A);
    }
    return solver.solve(b);
  }

  Real error() const override { return solver.error(); }

  [[nodiscard]] bool success() const override {
    return solver.info() == Eigen::Success;
  }
  static std::unique_ptr<LinearSolver> createFromJson(const core::JsonNode& json);

private:
  std::unique_ptr<LinearSolver> preconditioner;
  Eigen::ConjugateGradient<SparseMatrix<Real>> solver;
};
} // namespace maths