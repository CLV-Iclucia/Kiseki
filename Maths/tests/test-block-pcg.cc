#include <gtest/gtest.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Eigen/IterativeLinearSolvers>

using namespace ksk::maths;

// ============================================================
// Test 1: Diagonal SPD system
// A = diag([2I, 3I, 4I]), Jacobi precondition -> should converge in 1 iter
// ============================================================
TEST(BlockPCG, DiagonalSPDSystem) {
  BlockSparseMatrix<3> A(3, 3);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 2.0);
  A.addBlock(1, 1, I * 3.0);
  A.addBlock(2, 2, I * 4.0);

  // b = [(2,4,6), (9,12,15), (8,12,16)]
  BlockVector<3> b(3);
  b[0] = glm::dvec3(2.0, 4.0, 6.0);
  b[1] = glm::dvec3(9.0, 12.0, 15.0);
  b[2] = glm::dvec3(8.0, 12.0, 16.0);

  // Expected solution: x = [(1,2,3), (3,4,5), (2,3,4)]
  BlockVector<3> x(3);
  x.setZero();

  BlockPCGSolver solver;
  auto result = solver.solve(A, b, x);

  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.iterations, 2);  // Jacobi should make it converge in 1 iteration

  EXPECT_NEAR(x[0].x, 1.0, 1e-10);
  EXPECT_NEAR(x[0].y, 2.0, 1e-10);
  EXPECT_NEAR(x[0].z, 3.0, 1e-10);
  EXPECT_NEAR(x[1].x, 3.0, 1e-10);
  EXPECT_NEAR(x[1].y, 4.0, 1e-10);
  EXPECT_NEAR(x[1].z, 5.0, 1e-10);
  EXPECT_NEAR(x[2].x, 2.0, 1e-10);
  EXPECT_NEAR(x[2].y, 3.0, 1e-10);
  EXPECT_NEAR(x[2].z, 4.0, 1e-10);
}

// ============================================================
// Test 2: Coupled SPD system
// A = [[4I, I], [I, 4I]], verify solution matches Eigen CG (error < 1e-10)
// ============================================================
TEST(BlockPCG, CoupledSPDSystem) {
  BlockSparseMatrix<3> A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 4.0);
  A.addBlock(0, 1, I);
  A.addBlock(1, 0, I);
  A.addBlock(1, 1, I * 4.0);

  // x_true = [(1,2,3), (4,5,6)]
  BlockVector<3> x_true(2);
  x_true[0] = glm::dvec3(1.0, 2.0, 3.0);
  x_true[1] = glm::dvec3(4.0, 5.0, 6.0);

  // b = A * x_true
  BlockVector<3> b(2);
  A.apply(x_true, b);

  // Solve with Block PCG
  BlockVector<3> x(2);
  x.setZero();

  BlockPCGSolver solver(1000, 1e-10);
  auto result = solver.solve(A, b, x);

  EXPECT_TRUE(result.converged);

  // Compare with Eigen CG solution
  auto eigenA = A.toEigen();
  Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>> cg;
  cg.compute(eigenA);
  Eigen::VectorXd eigenX = cg.solve(b.asEigen());

  // Block PCG should match Eigen CG within tolerance
  auto diff = x.asEigen() - eigenX;
  EXPECT_LT(diff.norm(), 1e-10);

  // Also check against known solution
  EXPECT_NEAR(x[0].x, 1.0, 1e-10);
  EXPECT_NEAR(x[0].y, 2.0, 1e-10);
  EXPECT_NEAR(x[0].z, 3.0, 1e-10);
  EXPECT_NEAR(x[1].x, 4.0, 1e-10);
  EXPECT_NEAR(x[1].y, 5.0, 1e-10);
  EXPECT_NEAR(x[1].z, 6.0, 1e-10);
}

// ============================================================
// Test 3: fromEigen round-trip
// BlockSparseMatrix::fromEigen(A.toEigen()) should produce consistent apply() results
// ============================================================
TEST(BlockPCG, FromEigenRoundTrip) {
  BlockSparseMatrix<3> A(2, 2);
  glm::dmat3 I(1.0);
  glm::dmat3 M(0.0);
  M[0][0] = 1.0; M[1][0] = 0.5; M[2][0] = 0.0;
  M[0][1] = 0.5; M[1][1] = 2.0; M[2][1] = 0.3;
  M[0][2] = 0.0; M[1][2] = 0.3; M[2][2] = 3.0;
  A.addBlock(0, 0, M);
  A.addBlock(0, 1, I * 0.5);
  A.addBlock(1, 0, I * 0.5);
  A.addBlock(1, 1, M * 2.0);

  // Convert to Eigen and back
  auto eigenA = A.toEigen();
  auto A_roundtrip = BlockSparseMatrix<3>::fromEigen(eigenA);

  // Apply both to the same vector and compare
  BlockVector<3> x(2);
  x[0] = glm::dvec3(1.0, -2.0, 0.5);
  x[1] = glm::dvec3(-1.0, 3.0, 2.0);

  BlockVector<3> y_orig(2), y_roundtrip(2);
  A.apply(x, y_orig);
  A_roundtrip.apply(x, y_roundtrip);

  for (int i = 0; i < 2; i++) {
    EXPECT_NEAR(y_orig[i].x, y_roundtrip[i].x, 1e-14);
    EXPECT_NEAR(y_orig[i].y, y_roundtrip[i].y, 1e-14);
    EXPECT_NEAR(y_orig[i].z, y_roundtrip[i].z, 1e-14);
  }
}

// ============================================================
// Test 4: Non-zero initial guess reduces iterations
// ============================================================
TEST(BlockPCG, NonZeroInitialGuess) {
  BlockSparseMatrix<3> A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 4.0);
  A.addBlock(0, 1, I);
  A.addBlock(1, 0, I);
  A.addBlock(1, 1, I * 4.0);

  BlockVector<3> x_true(2);
  x_true[0] = glm::dvec3(1.0, 2.0, 3.0);
  x_true[1] = glm::dvec3(4.0, 5.0, 6.0);

  BlockVector<3> b(2);
  A.apply(x_true, b);

  // Solve with zero initial guess
  BlockVector<3> x_zero(2);
  x_zero.setZero();
  BlockPCGSolver solver1(1000, 1e-10);
  auto result_zero = solver1.solve(A, b, x_zero);

  // Solve with near-true initial guess
  BlockVector<3> x_warm(2);
  x_warm[0] = glm::dvec3(0.9, 1.9, 2.9);
  x_warm[1] = glm::dvec3(3.9, 4.9, 5.9);
  BlockPCGSolver solver2(1000, 1e-10);
  auto result_warm = solver2.solve(A, b, x_warm);

  EXPECT_TRUE(result_zero.converged);
  EXPECT_TRUE(result_warm.converged);
  // Warm start should converge in fewer iterations
  EXPECT_LE(result_warm.iterations, result_zero.iterations);
}

// ============================================================
// Test 5: Singular diagonal block handling (Jacobi regularization)
// ============================================================
TEST(BlockPCG, SingularDiagonalBlockNocrash) {
  // A has a singular (zero) diagonal block at position (1,1)
  // but we add off-diagonal coupling so the system is still solvable
  BlockSparseMatrix<3> A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 4.0);
  A.addBlock(0, 1, I * 0.1);
  A.addBlock(1, 0, I * 0.1);
  // Add a near-singular diagonal at (1,1)
  glm::dmat3 nearlySingular(0.0);
  nearlySingular[0][0] = 1e-15;
  nearlySingular[1][1] = 1e-15;
  nearlySingular[2][2] = 1e-15;
  A.addBlock(1, 1, nearlySingular);

  BlockVector<3> b(2);
  b[0] = glm::dvec3(1.0, 1.0, 1.0);
  b[1] = glm::dvec3(0.1, 0.1, 0.1);

  BlockVector<3> x(2);
  x.setZero();

  // Should not crash even with nearly singular diagonal
  BlockPCGSolver solver(100, 1e-4);
  auto result = solver.solve(A, b, x);

  // We don't require convergence here, just that it doesn't crash
  // The regularized Jacobi should handle it gracefully
  EXPECT_GE(result.iterations, 0);
}

// ============================================================
// Test 6: extractDiagonal correctness
// ============================================================
TEST(BlockSparseMatrix3Phase2A, ExtractDiagonal) {
  BlockSparseMatrix<3> A(3, 3);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 2.0);
  A.addBlock(0, 1, I * 0.5);
  A.addBlock(1, 1, I * 3.0);
  A.addBlock(2, 2, I * 4.0);
  // Add duplicate diagonal entry
  A.addBlock(1, 1, I * 1.0);  // should sum to 4I at (1,1)

  auto diag = A.extractDiagonal();
  ASSERT_EQ(diag.size(), 3u);

  for (int c = 0; c < 3; c++)
    for (int r = 0; r < 3; r++) {
      if (r == c) {
        EXPECT_DOUBLE_EQ(diag[0][c][r], 2.0);
        EXPECT_DOUBLE_EQ(diag[1][c][r], 4.0);  // 3 + 1
        EXPECT_DOUBLE_EQ(diag[2][c][r], 4.0);
      } else {
        EXPECT_DOUBLE_EQ(diag[0][c][r], 0.0);
        EXPECT_DOUBLE_EQ(diag[1][c][r], 0.0);
        EXPECT_DOUBLE_EQ(diag[2][c][r], 0.0);
      }
    }
}

// ============================================================
// Test 7: addFrom and scale
// ============================================================
TEST(BlockSparseMatrix3Phase2A, AddFromAndScale) {
  BlockSparseMatrix<3> A(2, 2), B(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 2.0);
  A.addBlock(1, 1, I * 3.0);
  B.addBlock(0, 0, I * 1.0);
  B.addBlock(0, 1, I * 0.5);
  B.addBlock(1, 1, I * 1.0);

  A.addFrom(B);

  BlockVector<3> x(2), y(2);
  x[0] = glm::dvec3(1.0, 0.0, 0.0);
  x[1] = glm::dvec3(0.0, 1.0, 0.0);

  A.apply(x, y);
  // y[0] = (2+1)*I*(1,0,0) + 0.5*I*(0,1,0) = (3, 0.5, 0)
  // y[1] = (3+1)*I*(0,1,0) = (0, 4, 0)
  EXPECT_NEAR(y[0].x, 3.0, 1e-14);
  EXPECT_NEAR(y[0].y, 0.5, 1e-14);
  EXPECT_NEAR(y[1].y, 4.0, 1e-14);

  // Test scale
  A.scale(2.0);
  A.apply(x, y);
  EXPECT_NEAR(y[0].x, 6.0, 1e-14);
  EXPECT_NEAR(y[0].y, 1.0, 1e-14);
  EXPECT_NEAR(y[1].y, 8.0, 1e-14);
}
