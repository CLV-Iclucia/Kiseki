// test_maths.cpp
#include <gtest/gtest.h>
#include <Maths/equations.h>

using namespace ksk::maths;

TEST(QuadraticSolveTest, RealRoots) {
  QuadraticPolynomial poly{1, -3, 2}; // x^2 - 3x + 2 = 0
  Real x1, x2;
  EXPECT_TRUE(quadraticSolve(poly, x1, x2));
  EXPECT_NEAR(x1, 2.0, 1e-9);
  EXPECT_NEAR(x2, 1.0, 1e-9);
}

TEST(QuadraticSolveTest, ComplexRoots) {
  QuadraticPolynomial poly{1, 2, 5}; // x^2 + 2x + 5 = 0, no real roots
  Real x1, x2;
  EXPECT_FALSE(quadraticSolve(poly, x1, x2));
}

TEST(CubicSolveTest, SingleRoot) {
  CubicPolynomial poly{1, 0, 0, -1}; // x^3 - 1 = 0
  CubicEquationRoots roots = clampedCubicSolve(poly, -10, 10, 1e-9);
  EXPECT_EQ(roots.numRoots, 1);
  EXPECT_NEAR(roots.roots[0], 1.0, 1e-9);
}

TEST(CubicSolveTest, TwoRoots) {
  CubicPolynomial poly{1, -4, 5, -2}; // (x-1)^2(x-2) = 0
  CubicEquationRoots roots = clampedCubicSolve(poly, -10, 10, 1e-9);
  EXPECT_EQ(roots.numRoots, 3);  // Two of the roots are the same
  EXPECT_NEAR(roots.roots[0], 1.0, 1e-7);
  EXPECT_NEAR(roots.roots[1], 1.0, 1e-7);
  EXPECT_NEAR(roots.roots[2], 2.0, 1e-7);
}

TEST(CubicSolveTest, ThreeRoots) {
  CubicPolynomial poly{1, -6, 11, -6}; // (x-1)(x-2)(x-3) = 0
  CubicEquationRoots roots = clampedCubicSolve(poly, -10, 10, 1e-9);
  EXPECT_EQ(roots.numRoots, 3);
  EXPECT_NEAR(roots.roots[0], 1.0, 1e-9);
  EXPECT_NEAR(roots.roots[1], 2.0, 1e-9);
  EXPECT_NEAR(roots.roots[2], 3.0, 1e-9);
}

TEST(BinaryLinearSolveTest, UniqueSolution) {
  BinaryLinearSystem sys{1, 2, 3, 4, 5, 6}; // 1x + 2y = 5, 3x + 4y = 6
  Real x, y;
  EXPECT_TRUE(binaryLinearSolve(sys, x, y));
  EXPECT_NEAR(x, -4.0, 1e-9);
  EXPECT_NEAR(y, 4.5, 1e-9);
}

TEST(BinaryLinearSolveTest, NoSolution) {
  BinaryLinearSystem sys{1, 2, 2, 4, 5, 10}; // No unique solution, lines are parallel
  Real x, y;
  EXPECT_FALSE(binaryLinearSolve(sys, x, y));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
