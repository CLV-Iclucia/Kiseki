// ============================================================================
// jacobi-eigen.hlsli — symmetric 9x9 eigen-decomposition + SPD filtering.
//
// Pure functions, no resource bindings — include into any kernel.
//
// Compile-time knobs (define before including, or via -D):
//   JACOBI_CLASSICAL   : if defined, pick largest off-diagonal each step;
//                        otherwise cyclic sweeps over all (p,q) pairs.
//   CLAMP_PSD          : if defined, eigenvalue clamp = max(lambda, 0);
//                        otherwise clamp = abs(lambda)  (matches CPU filter).
//   JACOBI_SWEEPS      : number of cyclic sweeps          (default 10)
//   JACOBI_ROTATIONS   : number of classical rotations    (default 400)
//
// Convergence is FIXED-iteration (no early-exit threshold) for deterministic,
// branch-light GPU execution. A tiny degenerate guard skips ~0 pivots.
//
// Matrices are dense double[9][9], symmetric on input.
// ============================================================================
#ifndef JACOBI_EIGEN_HLSLI
#define JACOBI_EIGEN_HLSLI

#ifndef JACOBI_SWEEPS
#define JACOBI_SWEEPS 10
#endif
#ifndef JACOBI_ROTATIONS
#define JACOBI_ROTATIONS 400
#endif

// Double-precision sqrt. DXC's `sqrt` intrinsic is float-only, so calling it on
// a double silently downcasts the argument and caps the whole eigensolver at
// float relative accuracy (~1e-7). We seed with the float sqrt and refine with
// two Newton iterations y <- 0.5*(y + x/y), which recover full double precision.
double dsqrt(double x) {
    if (x <= 0.0) return 0.0;
    double y = (double)sqrt((float)x); // ~float-accurate seed
    y = 0.5 * (y + x / y);             // Newton step 1
    y = 0.5 * (y + x / y);             // Newton step 2 -> full double
    return y;
}

// One symmetric Jacobi rotation zeroing A[p][q]; accumulates eigenvectors V.
// Uses the Numerical-Recipes stable form (tau / additive diagonal update) which
// avoids cancellation and converges in fewer sweeps than the c^2/s^2 form.
void jacobiRotate(inout double A[9][9], inout double V[9][9], int p, int q) {
    double apq = A[p][q];
    if (abs(apq) < 1e-300) return;

    double app = A[p][p], aqq = A[q][q];
    double theta = (aqq - app) / (2.0 * apq);
    double t   = (theta >= 0.0 ? 1.0 : -1.0) / (abs(theta) + dsqrt(theta * theta + 1.0));
    double c   = 1.0 / dsqrt(t * t + 1.0);
    double s   = t * c;
    double tau = s / (1.0 + c);

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0; A[q][p] = 0.0;

    [unroll] for (int k = 0; k < 9; ++k) {
        if (k != p && k != q) {
            double g = A[k][p], h = A[k][q];
            double npk = g - s * (h + tau * g);
            double nqk = h + s * (g - tau * h);
            A[k][p] = npk; A[p][k] = npk;
            A[k][q] = nqk; A[q][k] = nqk;
        }
    }
    [unroll] for (int k = 0; k < 9; ++k) {
        double g = V[k][p], h = V[k][q];
        V[k][p] = g - s * (h + tau * g);
        V[k][q] = h + s * (g - tau * h);
    }
}

// Eigen-decompose symmetric A in place: A becomes (near-)diagonal with the
// eigenvalues on its diagonal; V holds the eigenvectors as columns.
void jacobiEigen9(inout double A[9][9], inout double V[9][9]) {
    [unroll] for (int i = 0; i < 9; ++i)
        [unroll] for (int j = 0; j < 9; ++j)
            V[i][j] = (i == j) ? 1.0 : 0.0;

#ifdef JACOBI_CLASSICAL
    for (int it = 0; it < JACOBI_ROTATIONS; ++it) {
        int bp = 0, bq = 1; double best = 0.0;
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q) {
                double a = abs(A[p][q]);
                if (a > best) { best = a; bp = p; bq = q; }
            }
        if (best < 1e-300) break;  // degenerate guard, not a tolerance gate
        jacobiRotate(A, V, bp, bq);
    }
#else
    for (int sweep = 0; sweep < JACOBI_SWEEPS; ++sweep)
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q)
                jacobiRotate(A, V, p, q);
#endif
}

double clampEig(double lam) {
#ifdef CLAMP_PSD
    return max(lam, 0.0);
#else
    return abs(lam);
#endif
}

// In-place SPD filter: H <- V * diag(clamp(eigvals)) * V^T.
void filterSymmetric9(inout double H[9][9]) {
    double V[9][9];
    jacobiEigen9(H, V);                       // H diagonalized, V = eigenvectors

    double lam[9];
    [unroll] for (int i = 0; i < 9; ++i) lam[i] = clampEig(H[i][i]);

    [unroll] for (int i = 0; i < 9; ++i)
        [unroll] for (int j = 0; j < 9; ++j) {
            double sum = 0.0;
            [unroll] for (int k = 0; k < 9; ++k) sum += V[i][k] * lam[k] * V[j][k];
            H[i][j] = sum;
        }
}

#endif // JACOBI_EIGEN_HLSLI
