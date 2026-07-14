#include <Runtime/global-solver.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ksk::runtime {

void GlobalGaussNewtonSolver::prepare(const RuntimeScene& scene)
{
  scalar_count_ = scene.dofs.totalScalars;
  geometry_point_count_ = static_cast<int>(scene.geometry.points.size());
}

RuntimeStepResult GlobalGaussNewtonSolver::step(RuntimeSimulation& simulation,
                                                double dt,
                                                double time)
{
  prepare(simulation.scene());

  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  auto& subsystems = simulation.subsystems();

  for (const auto& subsystem : subsystems) {
    subsystem->readState(q, qdot);
    subsystem->beginStep(q, qdot, dt);
  }

  RuntimeStepResult result;
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;

  for (int iteration = 0; iteration < config.maxNewtonIterations; ++iteration) {
    DofBuffer gradient = DofBuffer::CPU(scalar_count);
    DofBuffer direction = DofBuffer::CPU(scalar_count);

    for (const auto& subsystem : subsystems) {
      subsystem->readState(q, qdot);
      subsystem->updateInternalConstraints(time + dt, dt);
      subsystem->prepareLocalOperator(dt);
      subsystem->assembleLocalGradient(gradient);
    }

    result.finalGradientNorm = gradient.norm();
    result.iterations = iteration + 1;
    if (result.finalGradientNorm <= config.newtonGradientTolerance) {
      result.converged = true;
      break;
    }

    if (!solveNewtonDirection(simulation, gradient, direction)) {
      result.finalStepNorm = 0.0;
      break;
    }

    const DofBuffer q_before = q.clone();
    const double objective_before = evaluateObjective(simulation, dt);
    const double directional_derivative = gradient.dot(direction);
    if (directional_derivative >= 0.0) {
      q.copyFrom(q_before);
      result.finalStepNorm = 0.0;
      break;
    }

    double alpha = 1.0;
    bool accepted = false;
    for (int line_search = 0;
         line_search < config.maxLineSearchIterations;
         ++line_search) {
      q.assignLinearCombination(q_before, alpha, direction);
      const double objective_trial = evaluateObjective(simulation, dt);
      const double armijo_rhs =
          objective_before +
          config.lineSearchArmijo * alpha * directional_derivative;
      if (std::isfinite(objective_trial) && objective_trial <= armijo_rhs) {
        accepted = true;
        break;
      }
      alpha *= config.lineSearchShrink;
    }

    if (!accepted) {
      q.copyFrom(q_before);
      result.finalStepNorm = 0.0;
      break;
    }

    result.finalStepNorm = alpha * direction.norm();
    if (result.finalStepNorm <= config.newtonStepTolerance) {
      result.converged = true;
      break;
    }
  }

  for (const auto& subsystem : subsystems) {
    subsystem->readState(q, qdot);
  }

  for (const auto& subsystem : subsystems) {
    subsystem->acceptStep(q, qdot, dt);
    subsystem->updateGeometry(simulation.scene().geometry);
  }

  return result;
}

double GlobalGaussNewtonSolver::evaluateObjective(
    RuntimeSimulation& simulation,
    double dt)
{
  double objective = 0.0;
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->readState(q, qdot);
    objective += subsystem->evaluateObjective(q, qdot, dt);
  }
  return objective;
}

bool GlobalGaussNewtonSolver::solveNewtonDirection(
    RuntimeSimulation& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction)
{
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;
  if (scalar_count == 0) {
    direction.setZero();
    return true;
  }

  if (simulation.subsystems().size() == 1 &&
      simulation.scene().contacts.stencils.empty()) {
    return solveSingleSubsystemDirection(simulation, gradient, direction);
  }

  direction.setZero();

  DofBuffer residual = DofBuffer::CPU(scalar_count);
  residual.assignScaled(-1.0, gradient);
  const double rhs_norm = residual.norm();
  if (rhs_norm == 0.0) {
    return true;
  }

  DofBuffer z = DofBuffer::CPU(scalar_count);
  applyPreconditioner(simulation, residual, z);

  DofBuffer search = z.clone();
  double rz_old = residual.dot(z);
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    return false;
  }

  const double tolerance = config.pcgTolerance * std::max(1.0, rhs_norm);
  DofBuffer matrix_search = DofBuffer::CPU(scalar_count);

  for (int iteration = 0; iteration < config.maxPcgIterations; ++iteration) {
    applyGlobalMatrix(simulation, search, matrix_search);

    const double denominator = search.dot(matrix_search);
    if (!std::isfinite(denominator) || denominator <= 0.0) {
      return false;
    }

    const double alpha = rz_old / denominator;
    direction.addScaled(alpha, search);
    residual.addScaled(-alpha, matrix_search);

    if (residual.norm() <= tolerance) {
      return true;
    }

    applyPreconditioner(simulation, residual, z);
    const double rz_new = residual.dot(z);
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      return false;
    }

    const double beta = rz_new / rz_old;
    search.assignLinearCombination(z, beta, search);
    rz_old = rz_new;
  }

  return residual.norm() <= tolerance;
}

bool GlobalGaussNewtonSolver::solveSingleSubsystemDirection(
    RuntimeSimulation& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction)
{
  direction.setZero();

  DofBuffer residual = DofBuffer::CPU(simulation.scene().dofs.totalScalars);
  residual.assignScaled(-1.0, gradient);
  if (residual.norm() == 0.0) {
    return true;
  }

  simulation.subsystems().front()->solveLocalSystem(residual, direction);
  return std::isfinite(direction.norm());
}

void GlobalGaussNewtonSolver::applyGlobalMatrix(RuntimeSimulation& simulation,
                                                const DofBuffer& x,
                                                DofBuffer& y)
{
  y.setZero();
  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->applyLocalMatrix(x, y);
  }

  const ContactTable& contacts = simulation.scene().contacts;
  if (contacts.stencils.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->applyContactHessian(x, contacts, y);
  }
}

void GlobalGaussNewtonSolver::applyPreconditioner(
    RuntimeSimulation& simulation,
    const DofBuffer& residual,
    DofBuffer& z)
{
  z.setZero();
  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->solveLocalSystem(residual, z);
  }
}

SimulationRunner::SimulationRunner(RuntimeSimulation simulation, double timeStep)
    : simulation_(std::move(simulation))
    , time_step_(timeStep)
{
  if (time_step_ <= 0.0) {
    throw std::invalid_argument("simulation time step must be positive");
  }
}

RuntimeStepResult SimulationRunner::step()
{
  last_step_ = solver_.step(simulation_, time_step_, time_);
  time_ += time_step_;
  ++steps_completed_;
  return last_step_;
}

RuntimeStepResult SimulationRunner::run(int steps)
{
  if (steps <= 0) {
    throw std::invalid_argument("simulation step count must be positive");
  }
  for (int step_index = 0; step_index < steps; ++step_index) {
    step();
  }
  return last_step_;
}

SimulationRunner buildSimulationRunner(const RuntimeSceneDesc& scene)
{
  return SimulationRunner(buildSimulation(scene), scene.timeStep);
}

}  // namespace ksk::runtime
