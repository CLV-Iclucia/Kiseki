#include <HairSim/implicit-integrator.h>

#include <Eigen/SparseCholesky>

#include <cmath>
#include <format>
#include <limits>
#include <utility>

namespace ksk::hairsim {
namespace {

Eigen::VectorXd flatten(const std::vector<RodBlock>& blocks) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(4 * blocks.size()));
  for (size_t i = 0; i < blocks.size(); ++i)
    for (int lane = 0; lane < 4; ++lane)
      result[static_cast<Eigen::Index>(4 * i + lane)] = blocks[i][lane];
  return result;
}

void assign(Eigen::VectorXd values, std::vector<RodBlock>& blocks) {
  for (size_t i = 0; i < blocks.size(); ++i)
    for (int lane = 0; lane < 4; ++lane)
      blocks[i][lane] = values[static_cast<Eigen::Index>(4 * i + lane)];
}

double implicitObjective(const Eigen::VectorXd& state,
                         const Eigen::VectorXd& predicted_state,
                         const Eigen::VectorXd& mass, double dt,
                         const RodEvaluation& evaluation) {
  const Eigen::VectorXd displacement = state - predicted_state;
  return evaluation.energy.total() +
         0.5 * (mass.array() * displacement.array().square()).sum() /
             (dt * dt);
}

}  // namespace

bool ImplicitRodIntegrator::step(double dt, const glm::dvec3& gravity) {
  diagnostic_.clear();
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    diagnostic_ = "time step must be finite and positive";
    return false;
  }

  const Eigen::VectorXd mass = rod_.massDiagonal();
  const RodState reference_previous_state = rod_.state();
  const Eigen::VectorXd previous_state = flatten(rod_.state().blocks);
  const Eigen::VectorXd old_velocity = flatten(rod_.velocity().blocks);
  const Eigen::VectorXd predicted_state = previous_state + dt * old_velocity;
  Eigen::VectorXd next_state = predicted_state;
  const Eigen::Index dummy = mass.size() - 1;
  next_state[dummy] = 0.0;

  static constexpr int MAX_NEWTON_ITERATIONS = 8;
  static constexpr int MAX_LINE_SEARCH_ITERATIONS = 12;
  static constexpr double FORCE_TOLERANCE = 1e-8;
  static constexpr double STEP_TOLERANCE = 1e-10;
  static constexpr double OBJECTIVE_TOLERANCE = 1e-12;
  for (int iteration = 0; iteration < MAX_NEWTON_ITERATIONS; ++iteration) {
    assign(next_state, rod_.state().blocks);
    last_evaluation_ = rod_.evaluate(gravity, reference_previous_state);
    if (!last_evaluation_.valid) {
      diagnostic_ = last_evaluation_.diagnostic;
      assign(previous_state, rod_.state().blocks);
      return false;
    }
    const double current_objective =
        implicitObjective(next_state, predicted_state, mass, dt,
                          last_evaluation_);
    if (!std::isfinite(current_objective)) {
      diagnostic_ = "implicit rod objective is not finite";
      assign(previous_state, rod_.state().blocks);
      assign(old_velocity, rod_.velocity().blocks);
      return false;
    }

    Eigen::VectorXd residual = flatten(last_evaluation_.gradient);
    residual += mass.asDiagonal() *
                ((next_state - predicted_state) / (dt * dt));
    residual[dummy] = next_state[dummy];
    if (residual.norm() < FORCE_TOLERANCE) break;

    Eigen::SparseMatrix<double> system = last_evaluation_.hessian;
    for (Eigen::Index i = 0; i < mass.size(); ++i)
      system.coeffRef(i, i) += mass[i] / (dt * dt);
    for (Eigen::SparseMatrix<double>::InnerIterator it(system, dummy); it; ++it)
      it.valueRef() = it.row() == dummy ? 1.0 : 0.0;
    for (Eigen::Index column = 0; column < system.outerSize(); ++column) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(system, column); it; ++it)
        if (it.row() == dummy && it.col() != dummy) it.valueRef() = 0.0;
    }
    system.coeffRef(dummy, dummy) = 1.0;
    system.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(system);
    if (solver.info() != Eigen::Success) {
      diagnostic_ = "implicit rod matrix factorization failed";
      assign(previous_state, rod_.state().blocks);
      return false;
    }

    const Eigen::VectorXd step = solver.solve(-residual);
    if (solver.info() != Eigen::Success || !step.allFinite()) {
      diagnostic_ = "implicit rod solve failed";
      assign(previous_state, rod_.state().blocks);
      return false;
    }
    if (step.norm() < STEP_TOLERANCE) break;

    bool accepted = false;
    double alpha = 1.0;
    double best_objective = std::numeric_limits<double>::infinity();
    double best_alpha = 0.0;
    std::string last_reject_reason = "no candidate evaluated";
    Eigen::VectorXd accepted_state = next_state;
    RodEvaluation accepted_evaluation;
    for (int line_search = 0; line_search < MAX_LINE_SEARCH_ITERATIONS;
         ++line_search) {
      Eigen::VectorXd candidate_state = next_state + alpha * step;
      candidate_state[dummy] = 0.0;
      assign(candidate_state, rod_.state().blocks);
      RodEvaluation candidate_evaluation =
          rod_.evaluate(gravity, reference_previous_state);
      if (candidate_evaluation.valid) {
        const double candidate_objective =
            implicitObjective(candidate_state, predicted_state, mass, dt,
                              candidate_evaluation);
        if (std::isfinite(candidate_objective) &&
            candidate_objective < best_objective) {
          best_objective = candidate_objective;
          best_alpha = alpha;
        }
        if (std::isfinite(candidate_objective) &&
            candidate_objective <=
                current_objective +
                    OBJECTIVE_TOLERANCE *
                        (1.0 + std::abs(current_objective))) {
          accepted = true;
          accepted_state = std::move(candidate_state);
          accepted_evaluation = std::move(candidate_evaluation);
          break;
        }
        last_reject_reason =
            std::isfinite(candidate_objective)
                ? "candidate did not decrease objective"
                : "candidate objective is not finite";
      } else {
        last_reject_reason = candidate_evaluation.diagnostic;
      }
      alpha *= 0.5;
    }

    if (!accepted) {
      diagnostic_ = std::format(
          "implicit rod line search failed: iteration={} current={:.17g} "
          "best={:.17g} best_alpha={:.3g} step_norm={:.17g} reason={}",
          iteration, current_objective, best_objective, best_alpha,
          step.norm(), last_reject_reason);
      assign(previous_state, rod_.state().blocks);
      assign(old_velocity, rod_.velocity().blocks);
      return false;
    }

    next_state = std::move(accepted_state);
    last_evaluation_ = std::move(accepted_evaluation);
    if ((alpha * step).norm() < STEP_TOLERANCE) break;
  }

  const Eigen::VectorXd next_velocity = (next_state - previous_state) / dt;
  assign(next_velocity, rod_.velocity().blocks);
  assign(next_state, rod_.state().blocks);
  rod_.transportReferenceFrames(reference_previous_state);
  last_evaluation_ = rod_.evaluate(gravity);
  if (!last_evaluation_.valid) {
    diagnostic_ = last_evaluation_.diagnostic;
    assign(previous_state, rod_.state().blocks);
    assign(old_velocity, rod_.velocity().blocks);
    return false;
  }
  return true;
}

}  // namespace ksk::hairsim
