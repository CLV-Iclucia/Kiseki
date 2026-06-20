#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Maths/types.h>
#include <Maths/block-vector.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace sim::maths {

/// Block Sparse Matrix in BCOO format, SOA layout.
///
/// Three parallel arrays:
///   m_blocks[k]   — the N×N glm block value
///   m_rowIdx[k]   — block row index
///   m_colIdx[k]   — block col index
///
/// SOA advantages:
///   - SpMV iterates m_blocks contiguously (cache-friendly for the hot data)
///   - Index arrays are narrow (int), separate cache lines
///   - Easier to SIMD/vectorize the block multiply loop
///
/// Duplicate (row, col) pairs allowed — summed during apply/toEigen.
template <int N>
class BlockSparseMatrix {
  static_assert(N >= 2 && N <= 4, "Block size must be 2, 3, or 4");

public:
  using Block = glm::mat<N, N, Real>;
  using Vec = glm::vec<N, Real>;

  BlockSparseMatrix() = default;
  BlockSparseMatrix(int blockRows, int blockCols)
      : m_blockRows(blockRows), m_blockCols(blockCols) {}

  // --- Dimensions ---
  [[nodiscard]] int blockRows() const { return m_blockRows; }
  [[nodiscard]] int blockCols() const { return m_blockCols; }
  [[nodiscard]] int scalarRows() const { return m_blockRows * N; }
  [[nodiscard]] int scalarCols() const { return m_blockCols * N; }
  [[nodiscard]] int numEntries() const { return static_cast<int>(m_blocks.size()); }

  // --- Assembly ---
  void setSize(int blockRows, int blockCols) {
    m_blockRows = blockRows;
    m_blockCols = blockCols;
  }

  void reserve(int nnzBlocks) {
    m_blocks.reserve(nnzBlocks);
    m_rowIdx.reserve(nnzBlocks);
    m_colIdx.reserve(nnzBlocks);
  }

  void addBlock(int blockRow, int blockCol, const Block &value) {
    assert(blockRow >= 0 && blockRow < m_blockRows);
    assert(blockCol >= 0 && blockCol < m_blockCols);
    m_blocks.push_back(value);
    m_rowIdx.push_back(blockRow);
    m_colIdx.push_back(blockCol);
  }

  void clear() {
    m_blocks.clear();
    m_rowIdx.clear();
    m_colIdx.clear();
    m_rowSegments.clear();
  }

  void clearAll() {
    m_blockRows = 0;
    m_blockCols = 0;
    clear();
  }

  // --- Raw access ---
  const std::vector<Block> &blocks() const { return m_blocks; }
  const std::vector<int> &rowIndices() const { return m_rowIdx; }
  const std::vector<int> &colIndices() const { return m_colIdx; }

  // --- Block SpMV: y = A * x ---
  void apply(const BlockVector<N> &x, BlockVector<N> &y) const {
    assert(x.numBlocks() == m_blockCols);
    assert(y.numBlocks() == m_blockRows);
    y.setZero();
    applyAdd(x, y);
  }

  /// y += A * x (accumulate, does not zero y)
  /// Uses parallel row-segmented approach when matrix is sorted by row.
  void applyAdd(const BlockVector<N> &x, BlockVector<N> &y) const {
    assert(x.numBlocks() == m_blockCols);
    assert(y.numBlocks() == m_blockRows);
    const int n = numEntries();

    // For small matrices or unsorted data, use serial scatter.
    if (n < 20000 || m_rowSegments.empty()) {
      for (int k = 0; k < n; k++) {
        y[m_rowIdx[k]] += m_blocks[k] * x[m_colIdx[k]];
      }
      return;
    }

    // Parallel path: each row-segment writes only to y[row], no conflict between rows.
    const int numSegments = static_cast<int>(m_rowSegments.size());
    tbb::parallel_for(0, numSegments, [&](int seg) {
      int segStart = m_rowSegments[seg];
      int segEnd = (seg + 1 < numSegments) ? m_rowSegments[seg + 1] : n;
      int row = m_rowIdx[segStart];
      Vec acc(0);
      for (int k = segStart; k < segEnd; k++) {
        acc += m_blocks[k] * x[m_colIdx[k]];
      }
      y[row] += acc;
    });
  }

  // --- Eigen bridge (copies data) ---
  [[nodiscard, deprecated("Use block operations directly")]] SparseMatrix<Real> toEigen() const {
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(numEntries() * N * N);
    const int n = numEntries();
    for (int k = 0; k < n; k++) {
      const int bRow = m_rowIdx[k];
      const int bCol = m_colIdx[k];
      const Block &block = m_blocks[k];
      for (int j = 0; j < N; j++)        // glm column
        for (int i = 0; i < N; i++)      // glm row
          triplets.emplace_back(bRow * N + i, bCol * N + j, block[j][i]);
    }
    SparseMatrix<Real> mat(scalarRows(), scalarCols());
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
  }

  // --- Block algebra ---

  /// Extract diagonal blocks. Duplicate (i,i) entries are summed.
  [[nodiscard]] std::vector<Block> extractDiagonal() const {
    std::vector<Block> diag(m_blockRows, Block(0));
    for (int k = 0; k < numEntries(); k++)
      if (m_rowIdx[k] == m_colIdx[k])
        diag[m_rowIdx[k]] += m_blocks[k];
    return diag;
  }

  /// Accumulate another same-sized BlockSparseMatrix: this += other
  void addFrom(const BlockSparseMatrix &other) {
    assert(m_blockRows == other.m_blockRows && m_blockCols == other.m_blockCols);
    for (int k = 0; k < other.numEntries(); k++)
      addBlock(other.m_rowIdx[k], other.m_colIdx[k], other.m_blocks[k]);
  }

  /// Scale all blocks by a scalar
  void scale(Real s) {
    for (auto &b : m_blocks)
      b *= s;
  }

  /// Accumulate another same-sized BlockSparseMatrix: this += other (syntax sugar for addFrom)
  BlockSparseMatrix &operator+=(const BlockSparseMatrix &other) {
    addFrom(other);
    return *this;
  }

  /// AXPY: this += a * other
  void axpy(Real a, const BlockSparseMatrix &other) {
    assert(m_blockRows == other.m_blockRows && m_blockCols == other.m_blockCols);
    for (int k = 0; k < other.numEntries(); k++)
      addBlock(other.m_rowIdx[k], other.m_colIdx[k], a * other.m_blocks[k]);
  }

  /// Assemble a local matrix (e.g., 12×12 = 4 blocks × 3 dof) into global matrix
  /// localMat is in Eigen row-major format, will be converted to glm column-major
  template <int LocalBlocks>
  void assembleBlock(const Eigen::Matrix<Real, LocalBlocks * 3, LocalBlocks * 3> &localMat,
                     const std::array<int, LocalBlocks> &blockIndices) {
    for (int i = 0; i < LocalBlocks; i++) {
      for (int j = 0; j < LocalBlocks; j++) {
        Block b;
        for (int c = 0; c < N; c++)       // col (glm column-major)
          for (int r = 0; r < N; r++)     // row
            b[c][r] = localMat(i * N + r, j * N + c);  // Eigen row-major -> glm col-major
        addBlock(blockIndices[i], blockIndices[j], b);
      }
    }
  }

  /// Add a diagonal scalar * Identity block at (blockIdx, blockIdx)
  void addDiagonalScalar(int blockIdx, Real scalar) {
    addBlock(blockIdx, blockIdx, Block(scalar));  // glm::dmat3(scalar) = scalar * I
  }

  // --- Utility ---
  /// Sort all arrays by (row, col) for better cache locality in apply().
  /// Also builds row-segment index for parallel SpMV.
  void sortByRow() {
    const int n = numEntries();
    if (n == 0) return;
    // Build permutation index
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
      return m_rowIdx[a] < m_rowIdx[b] ||
             (m_rowIdx[a] == m_rowIdx[b] && m_colIdx[a] < m_colIdx[b]);
    });
    // Apply permutation
    std::vector<Block> sortedBlocks(n);
    std::vector<int> sortedRow(n), sortedCol(n);
    for (int i = 0; i < n; i++) {
      sortedBlocks[i] = m_blocks[perm[i]];
      sortedRow[i] = m_rowIdx[perm[i]];
      sortedCol[i] = m_colIdx[perm[i]];
    }
    m_blocks = std::move(sortedBlocks);
    m_rowIdx = std::move(sortedRow);
    m_colIdx = std::move(sortedCol);

    // Build row-segment index: m_rowSegments[s] = start index of segment s
    m_rowSegments.clear();
    m_rowSegments.push_back(0);
    for (int k = 1; k < n; k++) {
      if (m_rowIdx[k] != m_rowIdx[k - 1])
        m_rowSegments.push_back(k);
    }
  }

private:
  int m_blockRows = 0;
  int m_blockCols = 0;
  std::vector<Block> m_blocks;
  std::vector<int> m_rowIdx;
  std::vector<int> m_colIdx;
  std::vector<int> m_rowSegments;  // Row-segment starts (built by sortByRow)
};

using BlockSparseMatrix3 = BlockSparseMatrix<3>;

} // namespace sim::maths
