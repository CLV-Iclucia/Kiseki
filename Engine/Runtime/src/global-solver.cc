#include <Runtime/global-solver.h>

#include <Runtime/contact-detection.h>
#include <Runtime/contact-barrier-energy.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                           int point_count)
{
  if (direction.isCPU())
    return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
}

DofBuffer createDofBufferLike(const DofBuffer& reference, int scalar_count)
{
  if (reference.isCPU()) {
    return DofBuffer::CPU(scalar_count);
  }
  return DofBuffer::GPU(*reference.device(), scalar_count);
}

}  // namespace

void GlobalGaussNewtonSolver::prepare(const RuntimeScene& scene)
{
  scalar_count_ = scene.dofs.totalScalars;
  geometry_point_count_ = static_cast<int>(scene.geometry.points.size());
}

RuntimeStepResult GlobalGaussNewtonSolver::step(SimulationContext& simulation,
                                                double dt,
                                                double time)
{
  prepare(simulation.scene());

  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  auto& subsystems = simulation.subsystems();

  for (const auto& subsystem : subsystems) {
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
    subsystem->beginStep(q.slice(range), qdot.slice(range), dt);
  }

  RuntimeStepResult result;
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;

  for (int iteration = 0; iteration < config.maxNewtonIterations; ++iteration) {
    DofBuffer gradient = createDofBufferLike(q, scalar_count);
    DofBuffer direction = createDofBufferLike(q, scalar_count);

    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
      subsystem->updateInternalConstraints(time + dt, dt);
      subsystem->prepareLocalOperator(dt);
      subsystem->assembleLocalGradient(gradient.slice(range));
      subsystem->updateGeometry(simulation.scene().geometry);
    }
    assembleContactGradient(simulation, gradient);

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
    updateContactsAlongDirection(simulation, direction);

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
         line_search++) {
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
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
  }

  for (const auto& subsystem : subsystems) {
    const DofRange range = subsystem->dofRange();
    subsystem->acceptStep(q.slice(range), qdot.slice(range), dt);
    subsystem->updateGeometry(simulation.scene().geometry);
  }

  return result;
}

double GlobalGaussNewtonSolver::evaluateObjective(
    SimulationContext& simulation,
    double dt)
{
  double objective = 0.0;
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
    objective +=
        subsystem->evaluateObjective(q.slice(range), qdot.slice(range), dt);
    subsystem->updateGeometry(simulation.scene().geometry);
  }
  objective += computeContactEnergy(simulation.scene().geometry, simulation.scene().contacts);
  return objective;
}

void GlobalGaussNewtonSolver::assembleContactGradient(
    SimulationContext& simulation,
    DofBuffer& gradient)
{
  const ContactStencils& contacts = simulation.scene().contacts;
  if (contacts.empty()) {
    return;
  }

  const ContactPotentialGradient contact_gradient =
      computeContactGradient(simulation.scene().geometry, contacts);
  if (contact_gradient.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    std::vector<PointIdx> points;
    std::vector<glm::dvec3> point_gradient;
    auto subsystem_id = subsystem->id();

    for (int i = 0; i < contact_gradient.points.size(); i++) {
      const PointIdx point = contact_gradient.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id) {
        continue;
      }
      points.push_back(point);
      point_gradient.push_back(contact_gradient.gradient.cpu()[i]);
    }

    if (points.empty()) {
      continue;
    }
    subsystem->scatterContactGradient(
        points,
        GeometryBuffer::FromCPU(std::move(point_gradient)).view().asConst(),
        gradient.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::updateContactsAlongDirection(
    SimulationContext& simulation,
    const DofBuffer& direction)
{
  RuntimeScene& scene = simulation.scene();
  const GlobalSolverConfig& solver_config = scene.solverConfig;
  const Real barrier_distance =
      solver_config.contactBarrierDistance > 0.0
          ? solver_config.contactBarrierDistance
          : solver_config.contactDetectionDistance;
  ContactDetectionConfig contact_config{
      .storage = solver_config.contactDetectionStorage,
      .detectionDistance = solver_config.contactDetectionDistance,
      .dHat = barrier_distance,
      .stiffness = solver_config.contactStiffness,
      .thickness = solver_config.contactThickness,
      .toi = 1.0,
  };

  GeometryBuffer geometry_direction =
      createGeometryDirectionBuffer(direction, scene.geometry.pointCount());

  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->mapLocalDirectionToGeometry(
        direction.slice(subsystem->dofRange()), geometry_direction.view());
  }

  GlobalContactRouter routed_contacts = runCCD(scene.geometry, geometry_direction, contact_config);
  scene.contacts = std::move(routed_contacts.globalContacts);

  for (const auto& subsystem : simulation.subsystems()) {
    ContactStencils contacts;
    const SubsystemId subsystem_id = subsystem->id();
    for (auto& entry : routed_contacts.subsystemInternalContacts | std::views::values) {
      if (entry.subsystem == subsystem_id) {
        contacts = std::move(entry.contacts);
        break;
      }
    }
    subsystem->applyInternalContacts(std::move(contacts));
  }
}

bool GlobalGaussNewtonSolver::solveNewtonDirection(
    SimulationContext& simulation,
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
      simulation.scene().contacts.empty()) {
    return solveSingleSubsystemDirection(simulation, gradient, direction);
  }

  direction.setZero();

  DofBuffer residual = createDofBufferLike(gradient, scalar_count);
  residual.assignScaled(-1.0, gradient);
  const double rhs_norm = residual.norm();
  if (rhs_norm == 0.0) {
    return true;
  }

  DofBuffer z = createDofBufferLike(gradient, scalar_count);
  applyPreconditioner(simulation, residual, z);

  DofBuffer search = z.clone();
  double rz_old = residual.dot(z);
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    return false;
  }

  const double tolerance = config.pcgTolerance * std::max(1.0, rhs_norm);
  DofBuffer matrix_search = createDofBufferLike(gradient, scalar_count);

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
    SimulationContext& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction)
{
  direction.setZero();

  DofBuffer residual =
      createDofBufferLike(gradient, simulation.scene().dofs.totalScalars);
  residual.assignScaled(-1.0, gradient);
  if (residual.norm() == 0.0) {
    return true;
  }

  const DofRange range = simulation.subsystems().front()->dofRange();
  simulation.subsystems().front()->solveLocalSystem(residual.slice(range),
                                                    direction.slice(range));
  return std::isfinite(direction.norm());
}

void GlobalGaussNewtonSolver::applyGlobalMatrix(SimulationContext& simulation,
                                                const DofBuffer& x,
                                                DofBuffer& y)
{
  y.setZero();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->applyLocalMatrix(x.slice(range), y.slice(range));
  }

  const ContactStencils& contacts = simulation.scene().contacts;
  if (!contacts.empty()) {
    applyGlobalContactHessian(simulation, x, y);
  }
  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->applyInternalContactHessian(
        x.slice(subsystem->dofRange()), y.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::applyGlobalContactHessian(
    SimulationContext& simulation,
    const DofBuffer& x,
    DofBuffer& y)
{
  const ContactStencils& contacts = simulation.scene().contacts;
  if (contacts.empty()) {
    return;
  }

  GeometryBuffer geometry_direction =
      createGeometryDirectionBuffer(x, simulation.scene().geometry.pointCount());
  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->mapLocalDirectionToGeometry(
        x.slice(subsystem->dofRange()), geometry_direction.view());
  }

  const ContactPotentialGradient contact_hessian_product =
      computeContactHessianProduct(
          simulation.scene().geometry,
          contacts,
          geometry_direction.view().asConst());
  if (contact_hessian_product.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    std::vector<PointIdx> points;
    std::vector<glm::dvec3> point_values;
    const SubsystemId subsystem_id = subsystem->id();

    for (int i = 0; i < contact_hessian_product.points.size(); ++i) {
      const PointIdx point = contact_hessian_product.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id)
        continue;
      points.push_back(point);
      point_values.push_back(contact_hessian_product.gradient.cpu()[i]);
    }

    if (points.empty()) {
      continue;
    }
    subsystem->scatterContactGradient(
        points,
        GeometryBuffer::FromCPU(std::move(point_values)).view().asConst(),
        y.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::applyPreconditioner(
    SimulationContext& simulation,
    const DofBuffer& residual,
    DofBuffer& z)
{
  z.setZero();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->solveLocalSystem(residual.slice(range), z.slice(range));
  }
}

SimulationRunner::SimulationRunner(SimulationContext simulation, double timeStep)
    : simulation_(std::move(simulation))
    , time_step_(timeStep)
{
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
  for (int step_index = 0; step_index < steps; ++step_index) {
    step();
  }
  return last_step_;
}

SimulationRunner buildSimulationRunner(const RuntimeSceneDesc& scene)
{
  return {buildSimulation(scene), scene.timeStep};
}

}  // namespace ksk::runtime
