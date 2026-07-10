#include <Maths/block-sparse-matrix.h>
#include <Maths/block-vector.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Maths/block-linear-solver.h>
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>

using namespace ksk::maths;
using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Utilities
// ============================================================

struct BenchmarkResult {
  std::string name;
  double median_ms;
  double min_ms;
  double max_ms;
  int iterations;
};

template <typename Func>
BenchmarkResult benchmark(const std::string &name, Func &&func, int warmup = 3, int runs = 10) {
  // Warmup
  for (int i = 0; i < warmup; i++)
    func();

  std::vector<double> times;
  times.reserve(runs);
  for (int i = 0; i < runs; i++) {
    auto start = Clock::now();
    func();
    auto end = Clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::sort(times.begin(), times.end());
  double median = times[runs / 2];
  double min_t = times.front();
  double max_t = times.back();

  return {name, median, min_t, max_t, runs};
}

void printResult(const BenchmarkResult &r) {
  std::cout << std::left << std::setw(40) << r.name
            << "  median=" << std::fixed << std::setprecision(3) << r.median_ms << " ms"
            << "  min=" << r.min_ms << " ms"
            << "  max=" << r.max_ms << " ms"
            << "  (n=" << r.iterations << ")"
            << std::endl;
}

// Generate a random SPD BlockSparseMatrix in BCOO format
// Simulates a FEM mesh connectivity pattern (each block-row has ~20 nonzeros)
BlockSparseMatrix<3> generateRandomSPDMatrix(int numBlocks, int avgNnzPerRow, std::mt19937 &rng) {
  BlockSparseMatrix<3> A(numBlocks, numBlocks);
  std::uniform_int_distribution<int> colDist(0, numBlocks - 1);
  std::uniform_real_distribution<Real> valDist(0.1, 2.0);

  // Diagonal dominance to ensure SPD
  for (int row = 0; row < numBlocks; row++) {
    Real diagSum = 0.0;
    int nnz = avgNnzPerRow;
    for (int k = 0; k < nnz; k++) {
      int col = colDist(rng);
      if (col == row) continue;
      glm::dmat3 block(0.0);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          block[j][i] = valDist(rng) * 0.1;  // small off-diag
      // Make symmetric: add both (row,col) and (col,row)
      A.addBlock(row, col, block);
      A.addBlock(col, row, glm::transpose(block));
      diagSum += 3.0 * 0.1;  // rough bound on spectral radius contribution
    }
    // Strong diagonal
    Real diagVal = diagSum + valDist(rng) * 10.0;
    A.addBlock(row, row, glm::dmat3(diagVal));
  }
  return A;
}

BlockVector<3> generateRandomVector(int numBlocks, std::mt19937 &rng) {
  BlockVector<3> v(numBlocks);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  for (int i = 0; i < numBlocks; i++)
    v[i] = glm::dvec3(dist(rng), dist(rng), dist(rng));
  return v;
}

// ============================================================
// Benchmarks
// ============================================================

void benchmarkSpMV(int numBlocks, int avgNnzPerRow) {
  std::mt19937 rng(42);
  auto A = generateRandomSPDMatrix(numBlocks, avgNnzPerRow, rng);
  auto x = generateRandomVector(numBlocks, rng);
  BlockVector<3> y(numBlocks);

  std::string label = "SpMV [" + std::to_string(numBlocks) + " blocks, "
                      + std::to_string(A.numEntries()) + " entries]";

  auto result = benchmark(label, [&]() {
    A.apply(x, y);
  });
  printResult(result);
}

void benchmarkDot(int numBlocks) {
  std::mt19937 rng(42);
  auto a = generateRandomVector(numBlocks, rng);
  auto b = generateRandomVector(numBlocks, rng);

  std::string label = "dot [" + std::to_string(numBlocks) + " blocks]";

  volatile Real sink = 0.0;
  auto result = benchmark(label, [&]() {
    sink = a.dot(b);
  });
  printResult(result);
}

void benchmarkAxpy(int numBlocks) {
  std::mt19937 rng(42);
  auto a = generateRandomVector(numBlocks, rng);
  auto b = generateRandomVector(numBlocks, rng);

  std::string label = "axpy [" + std::to_string(numBlocks) + " blocks]";

  auto result = benchmark(label, [&]() {
    a.axpy(0.5, b);
  });
  printResult(result);
}

void benchmarkNorm(int numBlocks) {
  std::mt19937 rng(42);
  auto a = generateRandomVector(numBlocks, rng);

  std::string label = "norm [" + std::to_string(numBlocks) + " blocks]";

  volatile Real sink = 0.0;
  auto result = benchmark(label, [&]() {
    sink = a.norm();
  });
  printResult(result);
}

void benchmarkBlockPCG(int numBlocks, int avgNnzPerRow) {
  std::mt19937 rng(42);
  auto A = generateRandomSPDMatrix(numBlocks, avgNnzPerRow, rng);
  A.sortByRow();
  auto b = generateRandomVector(numBlocks, rng);
  BlockVector<3> x(numBlocks);

  BlockPCGSolver solver(500, 1e-6);

  std::string label = "BlockPCG [" + std::to_string(numBlocks) + " blocks, "
                      + std::to_string(A.numEntries()) + " entries]";

  auto result = benchmark(label, [&]() {
    x.setZero();
    solver.solve(A, b, x);
  }, 1, 5);
  printResult(result);
}

void benchmarkJacobiSetup(int numBlocks, int avgNnzPerRow) {
  std::mt19937 rng(42);
  auto A = generateRandomSPDMatrix(numBlocks, avgNnzPerRow, rng);

  std::string label = "extractDiagonal [" + std::to_string(numBlocks) + " blocks, "
                      + std::to_string(A.numEntries()) + " entries]";

  auto result = benchmark(label, [&]() {
    auto diag = A.extractDiagonal();
    (void)diag;
  });
  printResult(result);
}

// ============================================================
// Main
// ============================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "  Kiseki Maths Benchmark (Baseline)   " << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  // --- BlockVector operations ---
  std::cout << "--- BlockVector Operations ---" << std::endl;
  for (int n : {1000, 5000, 20000}) {
    benchmarkDot(n);
    benchmarkAxpy(n);
    benchmarkNorm(n);
    std::cout << std::endl;
  }

  // --- SpMV ---
  std::cout << "--- SpMV (BCOO) ---" << std::endl;
  for (int n : {1000, 5000, 20000}) {
    benchmarkSpMV(n, 20);
  }
  std::cout << std::endl;

  // --- Jacobi Preconditioner Setup ---
  std::cout << "--- Jacobi Preconditioner Setup ---" << std::endl;
  for (int n : {1000, 5000, 20000}) {
    benchmarkJacobiSetup(n, 20);
  }
  std::cout << std::endl;

  // --- Block PCG (full solve) ---
  std::cout << "--- Block PCG (full solve) ---" << std::endl;
  for (int n : {1000, 5000}) {
    benchmarkBlockPCG(n, 20);
  }
  std::cout << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "  Benchmark complete.                   " << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
