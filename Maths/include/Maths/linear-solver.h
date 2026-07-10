//
// Created by CreeperIclucia-Vader on 25-5-16.
//

#pragma once

#include "types.h"
#include <Core/json.h>

namespace ksk::maths {

struct LinearSolver {
  virtual void reset() {};
  virtual Vector<Real, Dynamic> solve(const SparseMatrix<Real> &A,
                                      const Vector<Real, Dynamic> &b) = 0;
  [[nodiscard]] virtual Real error() const = 0;
  virtual bool success() const = 0;
  virtual ~LinearSolver() = default;
};

std::unique_ptr<LinearSolver> createLinearSolver(const core::JsonNode &node);

} // namespace maths
