#pragma once
#include <Maths/block-linear-solver.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string_view>

namespace ksk::maths {

/// Block PCG solver with built-in Block Jacobi preconditioning.
///
/// Default behavior: uses inline Block Jacobi (no virtual dispatch).
/// If a custom preconditioner is assigned, falls back to virtual dispatch.
struct BlockPCGSolver final : BlockLinearSolver {
  int maxIterations = 10000;
  Real tolerance = 1e-6;

  /// Optional custom preconditioner. If null, uses built-in Block Jacobi.
  std::unique_ptr<BlockPreconditioner> customPreconditioner = nullptr;

  BlockPCGSolver() = default;

  explicit BlockPCGSolver(int maxIter, Real tol)
      : maxIterations(maxIter), tolerance(tol) {}

  SolveResult solve(const BlockSparseMatrix<3>& A,
                    const BlockVector<3>& b,
                    BlockVector<3>& x) override {
    if (A.blockRows() != b.numBlocks() || A.blockCols() != x.numBlocks()) {
      throw std::runtime_error(std::format(
          "[BlockPCG] Dimension mismatch: A=({}x{} blocks), b={} blocks, x={} blocks",
          A.blockRows(), A.blockCols(), b.numBlocks(), x.numBlocks()));
    }

    requireFiniteVector("rhs b", b);
    requireFiniteVector("initial guess x", x);

    const int n = b.numBlocks();
    BlockVector<3> r(n), z(n), p(n), Ap(n);

    // Setup preconditioner
    if (customPreconditioner) {
      customPreconditioner->setup(A);
    } else {
      setupJacobi(A);
    }

    // r = b - A*x
    A.apply(x, r);
    requireFiniteVector("A*x", r);
    r *= Real(-1);
    r += b;
    requireFiniteVector("initial residual r", r);

    // z = M^{-1} r
    applyPrecond(r, z);
    requireFiniteVector("preconditioned residual z", z);
    p.copyFrom(z);
    requireFiniteVector("initial search direction p", p);

    Real rz = r.dot(z);
    requireFiniteScalar("initial rz", rz);

    Real b_norm = b.norm();
    requireFiniteScalar("rhs norm", b_norm);
    if (b_norm < 1e-30) return {true, 0, 0.0};

    for (int iter = 0; iter < maxIterations; iter++) {
      A.apply(p, Ap);
      requireFiniteVector(std::format("Ap at iter {}", iter), Ap);

      Real pAp = p.dot(Ap);
      requireFiniteScalar(std::format("pAp at iter {}", iter), pAp);
      if (std::abs(pAp) < 1e-30) {
        spdlog::warn("BlockPCG: pAp near zero at iter {}", iter);
        Real residual = r.norm();
        requireFiniteScalar(std::format("residual norm at iter {}", iter), residual);
        return {false, iter, residual};
      }

      Real alpha = rz / pAp;
      requireFiniteScalar(std::format("alpha at iter {}", iter), alpha);

      x.axpy(alpha, p);
      requireFiniteVector(std::format("solution x after iter {}", iter), x);
      r.axpy(-alpha, Ap);
      requireFiniteVector(std::format("residual r after iter {}", iter), r);

      Real r_norm = r.norm();
      requireFiniteScalar(std::format("residual norm after iter {}", iter), r_norm);
      if (r_norm / b_norm < tolerance)
        return {true, iter + 1, r_norm};

      applyPrecond(r, z);
      requireFiniteVector(std::format("preconditioned residual z after iter {}", iter), z);

      Real rz_new = r.dot(z);
      requireFiniteScalar(std::format("rz_new at iter {}", iter), rz_new);
      Real beta = rz_new / rz;
      requireFiniteScalar(std::format("beta at iter {}", iter), beta);
      rz = rz_new;

      p *= beta;
      p += z;
      requireFiniteVector(std::format("search direction p after iter {}", iter), p);
    }

    Real residual = r.norm();
    requireFiniteScalar("final residual norm", residual);
    return {false, maxIterations, residual};
  }

private:
  // --- Built-in Block Jacobi (no virtual dispatch) ---
  std::vector<glm::dmat3> m_invDiag;

  static bool isFiniteScalar(Real value) {
    return std::isfinite(value);
  }

  static bool isFiniteVec3(const glm::dvec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
  }

  static bool isFiniteMat3(const glm::dmat3& value) {
    for (int c = 0; c < 3; ++c)
      for (int r = 0; r < 3; ++r)
        if (!std::isfinite(value[c][r]))
          return false;
    return true;
  }

  static void requireFiniteScalar(std::string_view label, Real value) {
    if (!isFiniteScalar(value)) {
      throw std::runtime_error(
          std::format("[BlockPCG] {} is not finite: {}", label, value));
    }
  }

  static void requireFiniteVector(std::string_view label, const BlockVector<3>& value) {
    for (int i = 0; i < value.numBlocks(); ++i) {
      const auto& block = value[i];
      if (!isFiniteVec3(block)) {
        throw std::runtime_error(std::format(
            "[BlockPCG] {} has non-finite block {} = ({}, {}, {})",
            label, i, block.x, block.y, block.z));
      }
    }
  }

  void setupJacobi(const BlockSparseMatrix<3>& A) {
    auto diag = A.extractDiagonal();
    const int n = static_cast<int>(diag.size());
    m_invDiag.resize(n);

    const auto invertDiagBlock = [&](int i) {
      if (!isFiniteMat3(diag[i])) {
        throw std::runtime_error(std::format(
            "[BlockPCG] Jacobi diagonal block {} is not finite before inversion", i));
      }

      Real det = glm::determinant(diag[i]);
      requireFiniteScalar(std::format("Jacobi determinant at block {}", i), det);
      if (std::abs(det) < 1e-12)
        diag[i] += glm::dmat3(1e-10);

      m_invDiag[i] = glm::inverse(diag[i]);
      if (!isFiniteMat3(m_invDiag[i])) {
        throw std::runtime_error(std::format(
            "[BlockPCG] Inverted Jacobi block {} is not finite", i));
      }
    };

    if (n < 1000) {
      for (int i = 0; i < n; i++)
        invertDiagBlock(i);
    } else {
      tbb::parallel_for(tbb::blocked_range<int>(0, n),
          [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++)
              invertDiagBlock(i);
          });
    }
  }

  void applyPrecond(const BlockVector<3>& r, BlockVector<3>& z) const {
    if (customPreconditioner) {
      customPreconditioner->apply(r, z);
      requireFiniteVector("custom preconditioner output", z);
      return;
    }
    // Inline Block Jacobi: z[i] = invDiag[i] * r[i]
    const int n = r.numBlocks();
    if (n < 1000) {
      for (int i = 0; i < n; i++)
        z[i] = m_invDiag[i] * r[i];
    } else {
      tbb::parallel_for(tbb::blocked_range<int>(0, n),
          [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++)
              z[i] = m_invDiag[i] * r[i];
          });
    }
  }
};

} // namespace ksk::maths
