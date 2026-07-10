//
// Created by CreeperIclucia-Vader on 25-5-16.
//

#pragma once

#include <Maths/linear-solver.h>

namespace ksk::maths {
struct SparseCholeskySolver final : LinearSolver {
  Vector<Real, Dynamic> solve(const SparseMatrix<Real> &A, const Vector<Real, Dynamic> &b) override {
    solver.compute(A);
    return solver.solve(b);
  }

  Real error() const override {
    return 0.0;
  }

  [[nodiscard]] bool success() const override {
    return solver.info() == Eigen::Success;
  }
  static std::unique_ptr<LinearSolver> createFromJson(const core::JsonNode& json);

private:
  Eigen::SimplicialCholesky<SparseMatrix<Real>> solver;
};
}