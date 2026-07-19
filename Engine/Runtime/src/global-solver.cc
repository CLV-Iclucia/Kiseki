#include <Runtime/global-solver.h>

#include <Runtime/contact-detection.h>
#include <Runtime/contact-barrier-energy.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

void validateSimulationStateForStep(const SimulationContext& simulation)
{
  const RuntimeScene& scene = simulation.scene();
  const int scalar_count = scene.dofs.totalScalars;
  const int q_count = simulation.q().scalarCount();
  const int qdot_count = simulation.qdot().scalarCount();

  if (simulation.subsystems().empty()) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received a SimulationContext with no "
        "subsystems. Build it with buildSimulation/buildSimulationRunner "
        "from a populated RuntimeSceneDesc before stepping.");
  }
  if (scalar_count <= 0) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received a SimulationContext with zero "
        "DOFs. Check that scene objects were assigned to subsystems before "
        "buildSimulation/buildSimulationRunner.");
  }
  if (q_count != scalar_count || qdot_count != scalar_count) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received inconsistent DOF buffers: "
        "scene.dofs.totalScalars=" +
        std::to_string(scalar_count) +
        ", q.scalarCount=" +
        std::to_string(q_count) +
        ", qdot.scalarCount=" +
        std::to_string(qdot_count) + ".");
  }
}

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

ContactDetectionConfig createContactDetectionConfig(const RuntimeScene& scene,
                                                    Real toi)
{
  const GlobalSolverConfig& solver_config = scene.solverConfig;
  const Real barrier_distance =
      solver_config.contactBarrierDistance > 0.0
          ? solver_config.contactBarrierDistance
          : solver_config.contactDetectionDistance;
  return ContactDetectionConfig{
      .storage = solver_config.contactDetectionStorage,
      .detectionDistance = solver_config.contactDetectionDistance,
      .dHat = barrier_distance,
      .stiffness = solver_config.contactStiffness,
      .toi = toi,
  };
}

ContactDetectionConfig createCurrentBarrierContactConfig(
    const RuntimeScene& scene)
{
  ContactDetectionConfig config = createContactDetectionConfig(scene, 0.0);
  config.detectionDistance = config.dHat;
  return config;
}

void applyRoutedContacts(SimulationContext& simulation,
                         GlobalContactRouter routed_contacts)
{
  RuntimeScene& scene = simulation.scene();
  scene.contacts = std::move(routed_contacts.globalContacts);

  for (const auto& subsystem : simulation.subsystems()) {
    ContactStencils contacts;
    const auto found =
        routed_contacts.subsystemInternalContacts.find(subsystem->id());
    if (found != routed_contacts.subsystemInternalContacts.end()) {
      contacts = std::move(found->second.contacts);
    }
    subsystem->applyInternalContacts(std::move(contacts));
  }
}

void refreshContactsFromCandidates(SimulationContext& simulation,
                                   const ContactCandidates& candidates)
{
  GlobalContactRouter routed_contacts =
      refreshActiveContactsFromCandidates(
          simulation.scene().geometry,
          candidates,
          createCurrentBarrierContactConfig(simulation.scene()));
  applyRoutedContacts(simulation, std::move(routed_contacts));
}

void synchronizeGeometryFromState(SimulationContext& simulation)
{
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
    subsystem->updateGeometry(simulation.scene().geometry);
  }
}

ContactCandidates detectCandidatesWithGeometryDirection(
    SimulationContext& simulation,
    const GeometryBuffer& geometry_direction,
    const ContactDetectionConfig& contact_config)
{
  return detectContactCandidatesAlongDirection(
      simulation.scene().geometry, geometry_direction, contact_config);
}

void refreshContactsAtCurrentGeometry(SimulationContext& simulation)
{
  GeometryBuffer zero_direction =
      GeometryBuffer::CPU(simulation.scene().geometry.pointCount());
  const ContactCandidates candidates = detectCandidatesWithGeometryDirection(
      simulation,
      zero_direction,
      createCurrentBarrierContactConfig(simulation.scene()));
  refreshContactsFromCandidates(simulation, candidates);
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
  validateSimulationStateForStep(simulation);
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
      subsystem->updateGeometry(simulation.scene().geometry);
    }
    refreshContactsAtCurrentGeometry(simulation);

    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->prepareLocalOperator(dt);
      subsystem->assembleLocalGradient(gradient.slice(range));
    }
    assembleContactGradient(simulation, gradient);

    result.finalGradientNorm = gradient.norm();
    result.iterations = iteration + 1;
    if (result.finalGradientNorm <= config.newtonGradientTolerance) {
      result.converged = true;
      break;
    }

    const DofBuffer q_before = q.clone();
    const double objective_before = evaluateObjective(simulation, dt);

    if (!solveNewtonDirection(simulation, gradient, direction)) {
      result.finalStepNorm = 0.0;
      break;
    }
    const ContactSearchResult line_search_contacts =
        updateContactsAlongDirection(simulation, direction);

    const double directional_derivative = gradient.dot(direction);
    if (directional_derivative >= 0.0) {
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      refreshContactsAtCurrentGeometry(simulation);
      result.finalStepNorm = 0.0;
      break;
    }

    const double ccd_step_scale =
        std::clamp(config.ccdStepSizeScale, 0.0, 1.0);
    const double ccd_step_bound =
        std::clamp(line_search_contacts.stepSizeUpperBound, 0.0, 1.0);
    double alpha = ccd_step_bound;
    if (ccd_step_bound < 1.0) {
      alpha *= ccd_step_scale;
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      refreshContactsAtCurrentGeometry(simulation);
      result.finalStepNorm = 0.0;
      break;
    }

    bool accepted = false;
    for (int line_search = 0;
         line_search < config.maxLineSearchIterations;
         line_search++) {
      q.assignLinearCombination(q_before, alpha, direction);
      synchronizeGeometryFromState(simulation);
      refreshContactsFromCandidates(
          simulation, line_search_contacts.candidates);
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
      synchronizeGeometryFromState(simulation);
      refreshContactsAtCurrentGeometry(simulation);
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

GlobalGaussNewtonSolver::ContactSearchResult
GlobalGaussNewtonSolver::updateContactsAlongDirection(
    SimulationContext& simulation,
    const DofBuffer& direction)
{
  RuntimeScene& scene = simulation.scene();
  GeometryBuffer geometry_direction =
      createGeometryDirectionBuffer(direction, scene.geometry.pointCount());

  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->mapLocalDirectionToGeometry(
        direction.slice(subsystem->dofRange()), geometry_direction.view());
  }

  ContactCandidateDetectionResult detected =
      detectContactCandidatesAndStepSizeAlongDirection(
          scene.geometry,
          geometry_direction,
          createContactDetectionConfig(scene, 1.0));
  refreshContactsFromCandidates(simulation, detected.candidates);
  return ContactSearchResult{
      .candidates = std::move(detected.candidates),
      .stepSizeUpperBound = detected.stepSizeUpperBound,
  };
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
      computeContactHessianProductWrtGeometry(
          simulation.scene().geometry,
          contacts,
          geometry_direction.view().asConst());
  const ContactPotentialGradient contact_gradient =
      computeContactGradientWrtGeometry(simulation.scene().geometry, contacts);
  if (contact_hessian_product.points.empty() &&
      contact_gradient.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    std::vector<PointIdx> gradient_points;
    std::vector<glm::dvec3> gradient_values;
    std::vector<PointIdx> product_points;
    std::vector<glm::dvec3> product_values;
    const SubsystemId subsystem_id = subsystem->id();

    for (int i = 0; i < contact_gradient.points.size(); ++i) {
      const PointIdx point = contact_gradient.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id)
        continue;
      gradient_points.push_back(point);
      gradient_values.push_back(contact_gradient.gradient.cpu()[i]);
    }

    for (int i = 0; i < contact_hessian_product.points.size(); ++i) {
      const PointIdx point = contact_hessian_product.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id)
        continue;
      product_points.push_back(point);
      product_values.push_back(contact_hessian_product.gradient.cpu()[i]);
    }

    if (gradient_points.empty() && product_points.empty()) {
      continue;
    }
    subsystem->applyContactGeometryHessianProduct(
        gradient_points,
        GeometryBuffer::FromCPU(std::move(gradient_values)).view().asConst(),
        product_points,
        GeometryBuffer::FromCPU(std::move(product_values)).view().asConst(),
        x.slice(subsystem->dofRange()),
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
