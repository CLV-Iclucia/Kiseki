#pragma once

namespace ksk::runtime {

enum class RuntimeBackend {
  CPU,
  GPU,
};

struct SolverPlan {
  RuntimeBackend backend = RuntimeBackend::CPU;
  int maxNewtonIterations = 25;
  int maxPcgIterations = 200;
  double pcgTolerance = 1.0e-8;
};

}  // namespace ksk::runtime
