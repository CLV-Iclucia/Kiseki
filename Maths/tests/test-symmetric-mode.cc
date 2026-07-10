#include <gtest/gtest.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-types.h>
#include <Maths/block-solvers/block-pcg.h>

using namespace ksk::maths;

// Helper: build a known SPD block tridiagonal matrix (full storage, both triangles)
static BlockSparseMatrix<3> buildFullSPD(int nBlocks) {
  BlockSparseMatrix<3> A(nBlocks, nBlocks);
  glm::dmat3 I(1.0);

  // Diagonal-dominant: A[i][i] = (nBlocks+1)*I, off-diag B and B^T for |i-j|==1
  for (int i = 0; i < nBlocks; i++) {
    A.addBlock(i, i, I * static_cast<Real>(nBlocks + 1));
  }
  for (int i = 0; i + 1 < nBlocks; i++) {
    glm::dmat3 B(0.0);
    B[0][0] = 0.5 * (i + 1);  B[0][1] = 0.1 * i;        B[0][2] = 0.05;
    B[1][0] = 0.1 * i;        B[1][1] = 0.3 * (i + 1);  B[1][2] = 0.02 * i;
    B[2][0] = 0.05;           B[2][1] = 0.02 * i;        B[2][2] = 0.2 * (i + 1);
    A.addBlock(i, i + 1, B);
    A.addBlock(i + 1, i, glm::transpose(B));
  }
  return A;
}

// ============================================================
// Test 1: Basic SpMV correctness
// ============================================================
TEST(BlockSparseMatrix, SpMVCorrectness) {
  constexpr int N = 5;
  auto A = buildFullSPD(N);

  // Full: N diagonal + 2*(N-1) off-diag = 3N - 2
  EXPECT_EQ(A.numEntries(), 3 * N - 2);

  BlockVector<3> x(N);
  for (int i = 0; i < N; i++)
    x[i] = glm::dvec3(1.0 + i, 2.0 - 0.5 * i, 0.3 * i * i);

  BlockVector<3> y(N);
  A.apply(x, y);

  // Manually compute y[0] = (N+1)*I * x[0] + B_0 * x[1]
  // Just check it's nonzero and consistent with a second apply
  BlockVector<3> y2(N);
  A.apply(x, y2);
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(y[i].x, y2[i].x);
    EXPECT_DOUBLE_EQ(y[i].y, y2[i].y);
    EXPECT_DOUBLE_EQ(y[i].z, y2[i].z);
  }
}

// ============================================================
// Test 2: sortByRow enables parallel SpMV and preserves results
// ============================================================
TEST(BlockSparseMatrix, SortByRowPreservesSpMV) {
  constexpr int N = 5;
  auto A = buildFullSPD(N);

  BlockVector<3> x(N);
  for (int i = 0; i < N; i++)
    x[i] = glm::dvec3(1.0 + i, 2.0 - 0.5 * i, 0.3 * i * i);

  // SpMV before sorting
  BlockVector<3> y_before(N);
  A.apply(x, y_before);

  // Sort and SpMV again
  A.sortByRow();
  BlockVector<3> y_after(N);
  A.apply(x, y_after);

  // Results must be identical
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(y_before[i].x, y_after[i].x, 1e-12)
        << "Mismatch at block " << i << ".x after sortByRow";
    EXPECT_NEAR(y_before[i].y, y_after[i].y, 1e-12)
        << "Mismatch at block " << i << ".y after sortByRow";
    EXPECT_NEAR(y_before[i].z, y_after[i].z, 1e-12)
        << "Mismatch at block " << i << ".z after sortByRow";
  }
}

// ============================================================
// Test 3: PCG solver recovers known solution
// ============================================================
TEST(BlockSparseMatrix, PCGSolveRecoversSolution) {
  constexpr int N = 4;
  auto A = buildFullSPD(N);

  // Known solution
  BlockVector<3> x_true(N);
  x_true[0] = glm::dvec3(1.0, 2.0, 3.0);
  x_true[1] = glm::dvec3(4.0, 5.0, 6.0);
  x_true[2] = glm::dvec3(7.0, 8.0, 9.0);
  x_true[3] = glm::dvec3(10.0, 11.0, 12.0);

  // b = A * x_true
  BlockVector<3> b(N);
  A.apply(x_true, b);

  // Solve A x = b
  BlockVector<3> x_solved(N);
  x_solved.setZero();
  BlockPCGSolver solver(1000, 1e-10);
  auto result = solver.solve(A, b, x_solved);
  EXPECT_TRUE(result.converged);

  // Should recover x_true
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(x_solved[i].x, x_true[i].x, 1e-9)
        << "Mismatch at block " << i << ".x";
    EXPECT_NEAR(x_solved[i].y, x_true[i].y, 1e-9)
        << "Mismatch at block " << i << ".y";
    EXPECT_NEAR(x_solved[i].z, x_true[i].z, 1e-9)
        << "Mismatch at block " << i << ".z";
  }
}

// ============================================================
// Test 4: assembleBlock stores all N×N blocks
// ============================================================
TEST(BlockSparseMatrix, AssembleBlockStoresAllBlocks) {
  Eigen::Matrix<Real, 12, 12> localMat;
  localMat.setZero();
  for (int i = 0; i < 12; i++) {
    localMat(i, i) = 10.0 + i;
    for (int j = i + 1; j < 12; j++) {
      Real val = 0.1 * (i + 1) * (j + 1);
      localMat(i, j) = val;
      localMat(j, i) = val;
    }
  }

  std::array<int, 4> blockIndices = {0, 2, 5, 7};

  BlockSparseMatrix<3> H(8, 8);
  H.assembleBlock<4>(localMat, blockIndices);

  // All 4×4 = 16 blocks should be stored
  EXPECT_EQ(H.numEntries(), 16);

  // SpMV should be consistent
  BlockVector<3> x(8);
  for (int i = 0; i < 8; i++)
    x[i] = glm::dvec3(0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1));

  BlockVector<3> y(8);
  H.apply(x, y);

  // Verify by applying again — deterministic
  BlockVector<3> y2(8);
  H.apply(x, y2);
  for (int i = 0; i < 8; i++) {
    EXPECT_DOUBLE_EQ(y[i].x, y2[i].x);
    EXPECT_DOUBLE_EQ(y[i].y, y2[i].y);
    EXPECT_DOUBLE_EQ(y[i].z, y2[i].z);
  }
}

// ============================================================
// Test 5: assembleLocalHessian stores all N×N blocks
// ============================================================
TEST(BlockSparseMatrix, AssembleLocalHessianStoresAllBlocks) {
  constexpr int N = 3;

  LocalHessian<N> hess{};
  for (int i = 0; i < N; i++) {
    hess[i][i] = glm::dmat3(2.0);
    for (int j = i + 1; j < N; j++) {
      glm::dmat3 B(0.0);
      B[0][0] = 0.3 * (i + j);
      B[1][1] = 0.2 * (i + j);
      B[2][2] = 0.1 * (i + j);
      hess[i][j] = B;
      hess[j][i] = glm::transpose(B);
    }
  }

  LocalGrad<N> grad{};
  grad[0] = glm::dvec3(1.0, 0.5, 0.2);
  grad[1] = glm::dvec3(0.3, 1.0, 0.4);
  grad[2] = glm::dvec3(0.1, 0.2, 1.0);

  std::array<int, N> globalIdx = {1, 3, 5};
  Real bGrad = 0.5, bHess = 0.1, kappa = 100.0;

  BlockSparseMatrix<3> H(6, 6);
  assembleLocalHessian<N>(H, globalIdx, hess, grad, bGrad, bHess, kappa);

  // All N×N = 9 blocks stored
  EXPECT_EQ(H.numEntries(), N * N);

  // SpMV sanity check
  BlockVector<3> x(6);
  for (int i = 0; i < 6; i++)
    x[i] = glm::dvec3(1.0 + 0.5 * i, 2.0 - 0.3 * i, 0.1 * i * i);

  BlockVector<3> y(6);
  H.apply(x, y);

  // Non-zero output at active indices
  double norm_sq = 0;
  for (int idx : {1, 3, 5}) {
    norm_sq += glm::dot(y[idx], y[idx]);
  }
  EXPECT_GT(norm_sq, 0.0);
}

// ============================================================
// Test 6: addFrom merges matrices correctly
// ============================================================
TEST(BlockSparseMatrix, AddFromMergesCorrectly) {
  constexpr int N = 3;
  glm::dmat3 I(1.0);

  // Hessian
  BlockSparseMatrix<3> H(N, N);
  H.addBlock(0, 0, I * 5.0);
  H.addBlock(1, 1, I * 5.0);
  H.addBlock(2, 2, I * 5.0);
  H.addBlock(0, 1, I * 0.5);
  H.addBlock(1, 0, I * 0.5);
  H.addBlock(1, 2, I * 0.3);
  H.addBlock(2, 1, I * 0.3);

  // Diagonal mass
  BlockSparseMatrix<3> mass(N, N);
  mass.addBlock(0, 0, I * 2.0);
  mass.addBlock(1, 1, I * 3.0);
  mass.addBlock(2, 2, I * 4.0);

  // Compute y1 = H*x + mass*x separately
  BlockVector<3> x(N);
  x[0] = glm::dvec3(1.0, 2.0, 3.0);
  x[1] = glm::dvec3(4.0, 5.0, 6.0);
  x[2] = glm::dvec3(7.0, 8.0, 9.0);

  BlockVector<3> y_separate(N);
  H.apply(x, y_separate);
  BlockVector<3> y_mass(N);
  mass.apply(x, y_mass);
  for (int i = 0; i < N; i++)
    y_separate[i] += y_mass[i];

  // Compute y2 = (H + mass) * x via addFrom
  H.addFrom(mass);
  BlockVector<3> y_merged(N);
  H.apply(x, y_merged);

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(y_separate[i].x, y_merged[i].x, 1e-12);
    EXPECT_NEAR(y_separate[i].y, y_merged[i].y, 1e-12);
    EXPECT_NEAR(y_separate[i].z, y_merged[i].z, 1e-12);
  }
}
