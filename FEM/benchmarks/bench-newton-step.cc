#include <fem/system.h>
#include <fem/ipc/integrator.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <Maths/block-solvers/block-pcg.h>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace ksk;
using namespace ksk::fem;
using namespace ksk::maths;
using Clock = std::chrono::high_resolution_clock;

/// Simple timer for profiling individual operations
struct ScopedTimer {
  const char *label;
  std::chrono::time_point<Clock> start;

  explicit ScopedTimer(const char *label) : label(label), start(Clock::now()) {}
  ~ScopedTimer() {
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  [" << std::left << std::setw(30) << label << "]  "
              << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
  }
};

/// Build a simple test system from a .node/.ele tet mesh file pair
std::optional<System> buildTestSystem(const std::string &meshBasePath) {
  try {
    std::string nodePath = meshBasePath + ".node";
    std::string elePath = meshBasePath + ".ele";

    // Create elastic tet mesh primitive
    auto mesh = TetMesh::load(nodePath, elePath);
    if (!mesh) {
      std::cerr << "Failed to load mesh: " << meshBasePath << std::endl;
      return std::nullopt;
    }

    ElasticTetMesh elasticMesh;
    elasticMesh.setMesh(std::move(*mesh));
    elasticMesh.density = 1000.0;
    // StableNeoHookean with Young's=1e6, Poisson=0.4
    elasticMesh.energy = std::make_unique<deform::StableNeoHookean<Real>>(
        deform::ElasticityParameters<Real>{1e6, 0.4});

    System system;
    system.addPrimitive(Primitive(std::move(elasticMesh)));
    system.setGravity(glm::dvec3(0.0, -9.81, 0.0));
    system.init();

    return system;
  } catch (const std::exception &e) {
    std::cerr << "Error building system: " << e.what() << std::endl;
    return std::nullopt;
  }
}

void benchmarkHessianAssembly(System &system, int runs = 5) {
  std::cout << "\n--- Hessian Assembly ---" << std::endl;
  std::cout << "  System: " << system.numVertices() << " vertices, "
            << system.primitives().size() << " primitives" << std::endl;

  int numBlocks = system.x.numBlocks();
  BlockSparseMatrix<3> H(numBlocks, numBlocks);

  std::vector<double> times;
  for (int r = 0; r < runs; r++) {
    H.clear();
    auto start = Clock::now();
    system.spdProjectHessian(H);
    auto end = Clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::sort(times.begin(), times.end());
  std::cout << "  Hessian assembly: median=" << std::fixed << std::setprecision(3)
            << times[runs / 2] << " ms"
            << "  (nnz=" << H.numEntries() << " block entries)"
            << std::endl;
}

void benchmarkGradient(System &system, int runs = 5) {
  std::cout << "\n--- Gradient Computation ---" << std::endl;

  std::vector<double> times;
  for (int r = 0; r < runs; r++) {
    auto start = Clock::now();
    system.updateCurrentConfig(system.x);
    auto end = Clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::sort(times.begin(), times.end());
  std::cout << "  updateCurrentConfig (F + gradient): median=" << std::fixed
            << std::setprecision(3) << times[runs / 2] << " ms" << std::endl;
}

void benchmarkFullSolve(System &system) {
  std::cout << "\n--- Full Linear Solve (Hessian + PCG) ---" << std::endl;

  int numBlocks = system.x.numBlocks();
  BlockSparseMatrix<3> H(numBlocks, numBlocks);

  // Build Hessian
  system.spdProjectHessian(H);
  H.addFrom(system.blockMass());
  H.sortByRow();

  // Build RHS (use gradient as mock RHS)
  auto rhs = system.deformationEnergyGradient();

  // Solve
  BlockPCGSolver solver(1000, 1e-6);
  BlockVector<3> x(numBlocks);

  auto start = Clock::now();
  auto result = solver.solve(H, rhs, x);
  auto end = Clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "  PCG solve: " << ms << " ms"
            << "  (iters=" << result.iterations
            << ", converged=" << (result.converged ? "yes" : "NO")
            << ", residual=" << std::scientific << result.residualNorm << ")"
            << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "  Kiseki FEM Newton Step Benchmark    " << std::endl;
  std::cout << "========================================" << std::endl;

  // Try to load a mesh from FEM assets
  std::string meshPath;
  if (argc > 1) {
    meshPath = argv[1];
  } else {
    // Default: look for standard test meshes
    meshPath = std::string(FEM_TETS_DIR) + "/cube";
    std::cout << "Usage: bench-newton-step <mesh_base_path>" << std::endl;
    std::cout << "  (using default: " << meshPath << ")" << std::endl;
  }

  auto systemOpt = buildTestSystem(meshPath);
  if (!systemOpt) {
    std::cerr << "Failed to build test system. Exiting." << std::endl;
    return 1;
  }
  auto &system = *systemOpt;

  std::cout << "\nSystem info:" << std::endl;
  std::cout << "  Vertices: " << system.numVertices() << std::endl;
  std::cout << "  DOF: " << system.dof() << std::endl;
  std::cout << "  Triangles: " << system.numTriangles() << std::endl;
  std::cout << "  Edges: " << system.numEdges() << std::endl;

  // Run benchmarks
  benchmarkGradient(system);
  benchmarkHessianAssembly(system);
  benchmarkFullSolve(system);

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Benchmark complete.                   " << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
