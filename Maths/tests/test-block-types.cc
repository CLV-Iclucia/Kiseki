#include <gtest/gtest.h>
#include <Maths/block-vector.h>
#include <Maths/block-sparse-matrix.h>
#include <Eigen/IterativeLinearSolvers>

using namespace ksk::maths;

// ============================================================
// BlockVector tests
// ============================================================

TEST(BlockVector3, ConstructionAndAccess) {
  BlockVector3 v(4);
  EXPECT_EQ(v.numBlocks(), 4);
  EXPECT_EQ(v.scalarSize(), 12);

  v[0] = glm::dvec3(1.0, 2.0, 3.0);
  v[1] = glm::dvec3(4.0, 5.0, 6.0);

  EXPECT_DOUBLE_EQ(v[0].x, 1.0);
  EXPECT_DOUBLE_EQ(v[0].y, 2.0);
  EXPECT_DOUBLE_EQ(v[0].z, 3.0);
  EXPECT_DOUBLE_EQ(v[1].x, 4.0);
}

TEST(BlockVector3, DotProduct) {
  BlockVector3 a(2), b(2);
  a[0] = glm::dvec3(1.0, 0.0, 0.0);
  a[1] = glm::dvec3(0.0, 1.0, 0.0);
  b[0] = glm::dvec3(2.0, 3.0, 4.0);
  b[1] = glm::dvec3(5.0, 6.0, 7.0);

  // dot = 1*2 + 0*3 + 0*4 + 0*5 + 1*6 + 0*7 = 8
  EXPECT_DOUBLE_EQ(a.dot(b), 8.0);
}

TEST(BlockVector3, NormAndAxpy) {
  BlockVector3 v(1);
  v[0] = glm::dvec3(3.0, 4.0, 0.0);
  EXPECT_DOUBLE_EQ(v.norm(), 5.0);

  BlockVector3 w(1);
  w[0] = glm::dvec3(1.0, 1.0, 1.0);
  v.axpy(2.0, w);
  // v = (3+2, 4+2, 0+2) = (5, 6, 2)
  EXPECT_DOUBLE_EQ(v[0].x, 5.0);
  EXPECT_DOUBLE_EQ(v[0].y, 6.0);
  EXPECT_DOUBLE_EQ(v[0].z, 2.0);
}

TEST(BlockVector3, EigenBridgeZeroCopy) {
  BlockVector3 v(3);
  v[0] = glm::dvec3(1.0, 2.0, 3.0);
  v[1] = glm::dvec3(4.0, 5.0, 6.0);
  v[2] = glm::dvec3(7.0, 8.0, 9.0);

  auto eigenView = v.asEigen();
  EXPECT_EQ(eigenView.size(), 9);
  EXPECT_DOUBLE_EQ(eigenView(0), 1.0);
  EXPECT_DOUBLE_EQ(eigenView(3), 4.0);
  EXPECT_DOUBLE_EQ(eigenView(8), 9.0);

  // Modification through Eigen map visible in glm blocks
  eigenView(4) = 99.0;
  EXPECT_DOUBLE_EQ(v[1].y, 99.0);

  // Modification through glm block visible in Eigen map
  v[2].x = -1.0;
  EXPECT_DOUBLE_EQ(eigenView(6), -1.0);
}

TEST(BlockVector3, SetZero) {
  BlockVector3 v(2);
  v[0] = glm::dvec3(1.0, 2.0, 3.0);
  v[1] = glm::dvec3(4.0, 5.0, 6.0);
  v.setZero();
  EXPECT_DOUBLE_EQ(glm::length(v[0]), 0.0);
  EXPECT_DOUBLE_EQ(glm::length(v[1]), 0.0);
}

// ============================================================
// BlockSparseMatrix tests
// ============================================================

TEST(BlockSparseMatrix3, AssemblyAndDimensions) {
  BlockSparseMatrix3 A(2, 2);
  EXPECT_EQ(A.blockRows(), 2);
  EXPECT_EQ(A.blockCols(), 2);
  EXPECT_EQ(A.scalarRows(), 6);
  EXPECT_EQ(A.scalarCols(), 6);

  glm::dmat3 I(1.0);  // identity
  A.addBlock(0, 0, I * 2.0);
  A.addBlock(1, 1, I * 3.0);
  EXPECT_EQ(A.numEntries(), 2);

  A.clear();
  EXPECT_EQ(A.numEntries(), 0);
}

TEST(BlockSparseMatrix3, ApplyDiagonal) {
  // A = diag([2I, 3I]), x = [(1,1,1), (2,2,2)]
  // y = A*x = [(2,2,2), (6,6,6)]
  BlockSparseMatrix3 A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 2.0);
  A.addBlock(1, 1, I * 3.0);

  BlockVector3 x(2), y(2);
  x[0] = glm::dvec3(1.0, 1.0, 1.0);
  x[1] = glm::dvec3(2.0, 2.0, 2.0);

  A.apply(x, y);
  EXPECT_DOUBLE_EQ(y[0].x, 2.0);
  EXPECT_DOUBLE_EQ(y[0].y, 2.0);
  EXPECT_DOUBLE_EQ(y[0].z, 2.0);
  EXPECT_DOUBLE_EQ(y[1].x, 6.0);
  EXPECT_DOUBLE_EQ(y[1].y, 6.0);
  EXPECT_DOUBLE_EQ(y[1].z, 6.0);
}

TEST(BlockSparseMatrix3, ApplyOffDiagonal) {
  // A = [[I, 2I], [0, I]], x = [(1,0,0), (0,1,0)]
  // y[0] = I*(1,0,0) + 2I*(0,1,0) = (1,2,0)
  // y[1] = I*(0,1,0) = (0,1,0)
  BlockSparseMatrix3 A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I);
  A.addBlock(0, 1, I * 2.0);
  A.addBlock(1, 1, I);

  BlockVector3 x(2), y(2);
  x[0] = glm::dvec3(1.0, 0.0, 0.0);
  x[1] = glm::dvec3(0.0, 1.0, 0.0);

  A.apply(x, y);
  EXPECT_DOUBLE_EQ(y[0].x, 1.0);
  EXPECT_DOUBLE_EQ(y[0].y, 2.0);
  EXPECT_DOUBLE_EQ(y[0].z, 0.0);
  EXPECT_DOUBLE_EQ(y[1].x, 0.0);
  EXPECT_DOUBLE_EQ(y[1].y, 1.0);
  EXPECT_DOUBLE_EQ(y[1].z, 0.0);
}

TEST(BlockSparseMatrix3, DuplicateEntriesSum) {
  // Add identity to (0,0) twice → behaves like 2I
  BlockSparseMatrix3 A(1, 1);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I);
  A.addBlock(0, 0, I);

  BlockVector3 x(1), y(1);
  x[0] = glm::dvec3(1.0, 2.0, 3.0);

  A.apply(x, y);
  EXPECT_DOUBLE_EQ(y[0].x, 2.0);
  EXPECT_DOUBLE_EQ(y[0].y, 4.0);
  EXPECT_DOUBLE_EQ(y[0].z, 6.0);
}

TEST(BlockSparseMatrix3, ToEigen) {
  BlockSparseMatrix3 A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 4.0);
  A.addBlock(1, 1, I * 5.0);

  auto eigenA = A.toEigen();
  EXPECT_EQ(eigenA.rows(), 6);
  EXPECT_EQ(eigenA.cols(), 6);

  // Diagonal
  EXPECT_DOUBLE_EQ(eigenA.coeff(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(2, 2), 4.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(3, 3), 5.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(4, 4), 5.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(5, 5), 5.0);

  // Off-diagonal = 0
  EXPECT_DOUBLE_EQ(eigenA.coeff(0, 3), 0.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(3, 0), 0.0);
}

TEST(BlockSparseMatrix3, ToEigenNonDiagonalBlock) {
  // Verify a non-trivial (non-identity) block is correctly scattered
  BlockSparseMatrix3 A(1, 1);
  glm::dmat3 M(0.0);
  // M = [[1,2,3],[4,5,6],[7,8,9]]  (row-major intent)
  // glm is column-major: M[col][row]
  M[0][0] = 1.0; M[1][0] = 2.0; M[2][0] = 3.0;  // row 0
  M[0][1] = 4.0; M[1][1] = 5.0; M[2][1] = 6.0;  // row 1
  M[0][2] = 7.0; M[1][2] = 8.0; M[2][2] = 9.0;  // row 2
  A.addBlock(0, 0, M);

  auto eigenA = A.toEigen();
  EXPECT_DOUBLE_EQ(eigenA.coeff(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(eigenA.coeff(2, 2), 9.0);
}

TEST(BlockSparseMatrix3, RoundTripSolve) {
  // SPD block matrix: A = [[4I, I], [I, 4I]]
  BlockSparseMatrix3 A(2, 2);
  glm::dmat3 I(1.0);
  A.addBlock(0, 0, I * 4.0);
  A.addBlock(0, 1, I);
  A.addBlock(1, 0, I);
  A.addBlock(1, 1, I * 4.0);

  // x_true = [(1,2,3), (4,5,6)]
  BlockVector3 x_true(2);
  x_true[0] = glm::dvec3(1.0, 2.0, 3.0);
  x_true[1] = glm::dvec3(4.0, 5.0, 6.0);

  // b = A * x_true
  BlockVector3 b(2);
  A.apply(x_true, b);

  // Solve via Eigen bridge
  auto eigenA = A.toEigen();
  Eigen::ConjugateGradient<Eigen::SparseMatrix<Real>> cg;
  cg.compute(eigenA);
  Eigen::VectorXd eigenX = cg.solve(b.asEigen());

  // Verify solution matches
  auto diff = eigenX - x_true.asEigen();
  EXPECT_LT(diff.norm(), 1e-10);
}
