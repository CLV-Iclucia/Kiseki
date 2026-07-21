#pragma once

#include <Runtime/contact-detector.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/simulation-context.h>

#include <map>
#include <string>
#include <vector>

namespace ksk::runtime {

struct RuntimeStepResult {
  int iterations = 0;
  double finalGradientNorm = 0.0;
  double finalStepNorm = 0.0;
  bool converged = false;
};

class GlobalSolverStatsCollector {
 public:
  using StatMap = std::map<std::string, double>;

  void beginFrame();
  void clear() noexcept;

  void frameSet(std::string key, double value);
  void frameAdd(std::string key, double value);
  void frameMax(std::string key, double value);
  void frameMin(std::string key, double value);
  void frameAverage(std::string key, double value);

  void globalSet(std::string key, double value);
  void globalAdd(std::string key, double value);
  void globalMax(std::string key, double value);
  void globalMin(std::string key, double value);
  void globalAverage(std::string key, double value);

  [[nodiscard]] const StatMap& global() const noexcept { return global_; }
  [[nodiscard]] const std::vector<StatMap>& frames() const noexcept
  {
    return frames_;
  }

 private:
  static void add(StatMap& stats, std::string key, double value);
  static void max(StatMap& stats, std::string key, double value);
  static void min(StatMap& stats, std::string key, double value);
  static void average(StatMap& stats,
                      std::map<std::string, int>& counts,
                      std::string key,
                      double value);

  [[nodiscard]] StatMap& currentFrame();

  StatMap global_;
  std::map<std::string, int> global_average_counts_;
  std::vector<StatMap> frames_;
  std::vector<std::map<std::string, int>> frame_average_counts_;
};

class GlobalGaussNewtonSolver {
 public:
  void prepare(const RuntimeScene& scene);
  RuntimeStepResult step(SimulationContext& simulation,
                         double dt,
                         double time = 0.0,
                         GlobalSolverStatsCollector* stats = nullptr);

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
                                                 DofBuffer& direction,
                                                 GlobalSolverStatsCollector* stats);
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

  [[nodiscard]] RuntimeStepResult step(GlobalSolverStatsCollector* stats = nullptr);
  [[nodiscard]] RuntimeStepResult run(int steps,
                                      GlobalSolverStatsCollector* stats = nullptr);

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
