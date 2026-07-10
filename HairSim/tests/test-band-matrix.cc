#include <gtest/gtest.h>
#include <HairSim/band-matrix.h>
using hairksk::BandSquareMatrix;
using hairksk::BandLUSolver;
using hairksk::VecXd;

TEST(BandSquareMatrixTest, Assemble) {
  int n = 3;
  int bandwidth = 1;

  BandSquareMatrix<double, 1> A(n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j - i <= bandwidth && i - j <= bandwidth) {
        A(i, j) = i * n + j;
      }
    }
  }

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (j - i <= bandwidth && i - j <= bandwidth && i >= 0 && j >= 0)
        ASSERT_EQ(A(i, j), i * n + j);
}

// copied from stackoverflow
TEST(LAPACKTest, dgbsv) {
  int n = 10;
  // number of right-hand size
  int nrhs = 4;

  int ku = 2;
  int kl = 2;
  // ldab is larger than the number of bands,
  // to store the details of factorization
  int ldab = 2 * kl + ku + 1;

  //memory initialization
  double* a = (double*)malloc(n * ldab * sizeof(double));
  if (a == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  double* b = (double*)malloc(n * nrhs * sizeof(double));
  if (b == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  int* ipiv = (int*)malloc(n * sizeof(int));
  if (ipiv == NULL) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  int i, j;

  double fact = 1 * ((n + 1.) * (n + 1.));
  //matrix initialization : the different bands
  // are stored in rows kl <= j< 2kl+ku+1
  for (i = 0; i < n; i++) {
    a[(0 + kl) * n + i] = 0;
    a[(1 + kl) * n + i] = -1 * fact;
    a[(2 + kl) * n + i] = 2 * fact;
    a[(3 + kl) * n + i] = -1 * fact;
    a[(4 + kl) * n + i] = 0;

    //initialize source terms
    for (j = 0; j < nrhs; j++) {
      b[i * nrhs + j] = sin(M_PI * (i + 1) / (n + 1.));
    }
  }
  printf("end ini \n");

  int ierr;

  ierr = LAPACKE_dgbsv(LAPACK_ROW_MAJOR, n, kl, ku, nrhs, a, n, ipiv, b, nrhs);

  if (ierr < 0) { LAPACKE_xerbla("LAPACKE_dgbsv", ierr); }

  printf("output of LAPACKE_dgbsv\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < nrhs; j++) {
      printf("%g ", b[i * nrhs + j]);
    }
    printf("\n");
  }

  //checking correctness
  double norm = 0;
  double diffnorm = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < nrhs; j++) {
      norm += b[i * nrhs + j] * b[i * nrhs + j];
      diffnorm += (b[i * nrhs + j] - 1. / (M_PI * M_PI) *
                   sin(M_PI * (i + 1) / (n + 1.))) * (
        b[i * nrhs + j] - 1. / (M_PI * M_PI) * sin(M_PI * (i + 1) / (n + 1.)));
    }
  }
  printf("analical solution is 1/(PI*PI)*sin(x)\n");
  printf("relative difference is %g\n", sqrt(diffnorm / norm));

  free(a);
  free(b);
  free(ipiv);
}

TEST(BandLUSolverTest, SolveLinearSystem) {
  int n = 10;

  BandSquareMatrix<double, 1> A(n);

  A(0, 0) = 2;
  A(0, 1) = -1;
  A(1, 0) = -1;
  A(1, 1) = 2;
  A(1, 2) = -1;
  A(2, 1) = -1;
  A(2, 2) = 2;
  A(2, 3) = -1;
  A(3, 2) = -1;
  A(3, 3) = 2;
  A(3, 4) = -1;
  A(4, 3) = -1;
  A(4, 4) = 2;
  A(4, 5) = -1;
  A(5, 4) = -1;
  A(5, 5) = 2;
  A(5, 6) = -1;
  A(6, 5) = -1;
  A(6, 6) = 2;
  A(6, 7) = -1;
  A(7, 6) = -1;
  A(7, 7) = 2;
  A(7, 8) = -1;
  A(8, 7) = -1;
  A(8, 8) = 2;
  A(8, 9) = -1;
  A(9, 8) = -1;
  A(9, 9) = 2;

  VecXd rhs(n);
  rhs << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  VecXd x(n);

  BandLUSolver<double, 1> solver;
  solver.solve(A, rhs, x);
  ASSERT_TRUE(solver.success());
  for (int i = 0; i < n; ++i)
    std::cout << x(i) << std::endl;
  ASSERT_EQ(x.size(), n);
  ASSERT_NEAR(x(0), 1.0, 1e-6);
  ASSERT_NEAR(x(1), 1.0, 1e-6);
  ASSERT_NEAR(x(2), 1.0, 1e-6);
  ASSERT_NEAR(x(3), 1.0, 1e-6);
  ASSERT_NEAR(x(4), 1.0, 1e-6);
  ASSERT_NEAR(x(5), 1.0, 1e-6);
  ASSERT_NEAR(x(6), 1.0, 1e-6);
  ASSERT_NEAR(x(7), 1.0, 1e-6);
  ASSERT_NEAR(x(8), 1.0, 1e-6);
  ASSERT_NEAR(x(9), 1.0, 1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}