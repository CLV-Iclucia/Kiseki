#pragma once
#include <Maths/block-linear-solver.h>
#include <glm/glm.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace ksk::maths {

struct BlockJacobiPreconditioner final : BlockPreconditioner {
  void setup(const BlockSparseMatrix<3>& A) override {
    auto diag = A.extractDiagonal();
    m_invDiag.resize(diag.size());
    const int n = static_cast<int>(diag.size());

    if (n < 1000) {
      // Serial path for small systems
      for (int i = 0; i < n; i++) {
        Real det = glm::determinant(diag[i]);
        if (std::abs(det) < 1e-12)
          diag[i] += glm::dmat3(1e-10);
        m_invDiag[i] = glm::inverse(diag[i]);
      }
    } else {
      // Parallel path: each 3x3 inverse is independent
      tbb::parallel_for(tbb::blocked_range<int>(0, n),
          [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++) {
              Real det = glm::determinant(diag[i]);
              if (std::abs(det) < 1e-12)
                diag[i] += glm::dmat3(1e-10);
              m_invDiag[i] = glm::inverse(diag[i]);
            }
          });
    }
  }

  void apply(const BlockVector<3>& r, BlockVector<3>& z) const override {
    assert(r.numBlocks() == static_cast<int>(m_invDiag.size()));
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

private:
  std::vector<glm::dmat3> m_invDiag;
};

} // namespace ksk::maths
