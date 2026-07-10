#pragma once
#include <Maths/block-vector.h>
#include <Maths/block-sparse-matrix.h>
#include <memory>

namespace ksk::maths {

struct SolveResult {
  bool converged = false;
  int iterations = 0;
  Real residualNorm = 0.0;
};

/// Block preconditioner interface (non-template, one overload per block size)
struct BlockPreconditioner {
  virtual void setup(const BlockSparseMatrix<3>& A) = 0;
  virtual void apply(const BlockVector<3>& r, BlockVector<3>& z) const = 0;
  virtual ~BlockPreconditioner() = default;
};

/// Block linear solver interface (non-template)
struct BlockLinearSolver {
  virtual SolveResult solve(const BlockSparseMatrix<3>& A,
                            const BlockVector<3>& b,
                            BlockVector<3>& x) = 0;
  virtual ~BlockLinearSolver() = default;
};

} // namespace ksk::maths
