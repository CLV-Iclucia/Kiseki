#pragma once

#include <Runtime/contact-detector.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/simulation-context.h>

namespace ksk::runtime {

struct RuntimeStepResult {
  int iterations = 0;
  double finalGradientNorm = 0.0;
  double finalStepNorm = 0.0;
  bool converged = false;
};

class GlobalGaussNewtonSolver {
 public:
  void prepare(const RuntimeScene& scene);
  RuntimeStepResult step(SimulationContext& simulation,
                         double dt,
                         double time = 0.0);

  [[nodiscard]] int scalarCount() const noexcept { return scalar_count_; }
  [[nodiscard]] int geometryPointCount() const noexcept
  {
    return geometry_point_count_;
  }

 private:
  [[nodiscard]] static double evaluateObjective(SimulationContext& simulation,
                                                double dt);
  static void assembleContactGradient(SimulationContext& simulation,
                                      DofBuffer& gradient);
  [[nodiscard]] static bool solveNewtonDirection(SimulationContext& simulation,
                                                 const DofBuffer& gradient,
                                                 DofBuffer& direction);
  [[nodiscard]] static bool solveSingleSubsystemDirection(
      SimulationContext& simulation,
      const DofBuffer& gradient,
      DofBuffer& direction);
  static void applyGlobalMatrix(SimulationContext& simulation,
                                const DofBuffer& x,
                                DofBuffer& y);
  static void applyGlobalContactHessian(SimulationContext& simulation,
                                        const DofBuffer& x,
                                        DofBuffer& y);
  static void applyPreconditioner(SimulationContext& simulation,
                                  const DofBuffer& residual,
                                  DofBuffer& z);

  int scalar_count_ = 0;
  int geometry_point_count_ = 0;
  ContactDetector contact_detector_;
};

class SimulationRunner {
 public:
  SimulationRunner() = default;
  SimulationRunner(SimulationContext simulation, double timeStep);

  [[nodiscard]] RuntimeStepResult step();
  [[nodiscard]] RuntimeStepResult run(int steps);

  [[nodiscard]] SimulationContext& simulation() noexcept { return simulation_; }
  [[nodiscard]] const SimulationContext& simulation() const noexcept
  {
    return simulation_;
  }
  [[nodiscard]] const RuntimeStepResult& lastStep() const noexcept
  {
    return last_step_;
  }
  [[nodiscard]] double time() const noexcept { return time_; }
  [[nodiscard]] double timeStep() const noexcept { return time_step_; }
  [[nodiscard]] int stepsCompleted() const noexcept { return steps_completed_; }

 private:
  SimulationContext simulation_;
  GlobalGaussNewtonSolver solver_;
  RuntimeStepResult last_step_;
  double time_step_ = 1.0 / 60.0;
  double time_ = 0.0;
  int steps_completed_ = 0;
};

[[nodiscard]] SimulationRunner buildSimulationRunner(
    const RuntimeSceneDesc& scene);

}  // namespace ksk::runtime
