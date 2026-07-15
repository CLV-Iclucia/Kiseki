#include <FEM/cpu/fem-cpu-backend.h>

#include <FEM/fem-subsystem.h>

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::engine::fem {
namespace {

void ensureWritable(Eigen::VectorXd& values, int minimumSize)
{
  if (values.size() >= minimumSize) {
    return;
  }
  const Eigen::Index old_size = values.size();
  values.conservativeResize(minimumSize);
  values.segment(old_size, minimumSize - old_size).setZero();
}

void requireCPU(const runtime::DofBuffer& buffer, const char* operation)
{
  if (!buffer.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU DofBuffer");
  }
}

void requireCPU(const runtime::GeometryBuffer& buffer, const char* operation)
{
  if (!buffer.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU GeometryBuffer");
  }
}

}  // namespace

FEMCPUBackend::FEMCPUBackend(FEMSubsystem& subsystem)
    : subsystem_(subsystem)
    , mass_diagonal_(buildMassDiagonal())
{
}

void FEMCPUBackend::writeState(runtime::DofBuffer& q,
                               runtime::DofBuffer& qdot) const
{
  requireCPU(q, "FEMCPUBackend::writeState");
  requireCPU(qdot, "FEMCPUBackend::writeState");

  const runtime::DofRange range = subsystem_.range_;
  Eigen::VectorXd& q_values = q.cpu();
  Eigen::VectorXd& qdot_values = qdot.cpu();
  ensureWritable(q_values, range.scalarOffset + range.scalarCount);
  ensureWritable(qdot_values, range.scalarOffset + range.scalarCount);

  for (int mesh_index = 0;
       mesh_index < static_cast<int>(subsystem_.meshes_.size());
       ++mesh_index) {
    const TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
    const int base = range.scalarOffset + subsystem_.mesh_offsets_[mesh_index].q;
    for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
         ++vertex) {
      const glm::dvec3 position =
          subsystem_.vertexPosition(mesh_index, vertex);
      const glm::dvec3 velocity =
          subsystem_.vertexVelocity(mesh_index, vertex);
      q_values[base + 3 * vertex + 0] = position.x;
      q_values[base + 3 * vertex + 1] = position.y;
      q_values[base + 3 * vertex + 2] = position.z;
      qdot_values[base + 3 * vertex + 0] = velocity.x;
      qdot_values[base + 3 * vertex + 1] = velocity.y;
      qdot_values[base + 3 * vertex + 2] = velocity.z;
    }
  }
}

void FEMCPUBackend::readState(runtime::DofBuffer& q,
                              runtime::DofBuffer& qdot)
{
  requireCPU(q, "FEMCPUBackend::readState");
  requireCPU(qdot, "FEMCPUBackend::readState");

  ensureWritable(q.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);
  ensureWritable(qdot.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);

  for (int mesh_index = 0;
       mesh_index < static_cast<int>(subsystem_.meshes_.size());
       ++mesh_index) {
    TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
    const int base = subsystem_.mesh_offsets_[mesh_index].q;
    if (mesh.initialPositions.empty()) {
      mesh.initialPositions = mesh.vertices;
    }
    if (mesh.initialVelocities.empty()) {
      mesh.initialVelocities.resize(mesh.vertices.size(), glm::dvec3(0.0));
    }
    for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
         ++vertex) {
      subsystem_.setVertexPosition(
          mesh_index, vertex,
          glm::dvec3(
              q.cpu()[subsystem_.vectorOffset(q.cpu(), base + 3 * vertex + 0)],
              q.cpu()[subsystem_.vectorOffset(q.cpu(), base + 3 * vertex + 1)],
              q.cpu()[subsystem_.vectorOffset(q.cpu(), base + 3 * vertex + 2)]));
      subsystem_.setVertexVelocity(
          mesh_index, vertex,
          glm::dvec3(qdot.cpu()[subsystem_.vectorOffset(
                         qdot.cpu(), base + 3 * vertex + 0)],
                     qdot.cpu()[subsystem_.vectorOffset(
                         qdot.cpu(), base + 3 * vertex + 1)],
                     qdot.cpu()[subsystem_.vectorOffset(
                         qdot.cpu(), base + 3 * vertex + 2)]));
    }
  }
}

void FEMCPUBackend::beginStep(const runtime::DofBuffer& q,
                              const runtime::DofBuffer& qdot,
                              double dt)
{
  requireCPU(q, "FEMCPUBackend::beginStep");
  requireCPU(qdot, "FEMCPUBackend::beginStep");
  step_dt_ = dt;
  step_start_ = subsystem_.gatherLocalVector(q.cpu());
  inertial_target_ =
      step_start_ + dt * subsystem_.gatherLocalVector(qdot.cpu());
  mass_diagonal_ = buildMassDiagonal();
}

void FEMCPUBackend::acceptStep(const runtime::DofBuffer& q,
                               runtime::DofBuffer& qdot,
                               double dt)
{
  requireCPU(q, "FEMCPUBackend::acceptStep");
  requireCPU(qdot, "FEMCPUBackend::acceptStep");
  if (step_start_.size() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend step has not begun");
  }

  const Eigen::VectorXd local_q = subsystem_.gatherLocalVector(q.cpu());
  const Eigen::VectorXd local_qdot = (local_q - step_start_) / dt;
  ensureWritable(qdot.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    qdot.cpu()[subsystem_.vectorOffset(qdot.cpu(), i)] = local_qdot[i];
  }
}

double FEMCPUBackend::evaluateObjective(const runtime::DofBuffer& q,
                                        const runtime::DofBuffer& qdot,
                                        double dt)
{
  (void)qdot;
  requireCPU(q, "FEMCPUBackend::evaluateObjective");
  const Eigen::VectorXd local_q = subsystem_.gatherLocalVector(q.cpu());
  const Eigen::VectorXd residual = local_q - inertial_target_;
  double objective =
      0.5 *
      (mass_diagonal_.array() * residual.array().square()).sum() /
      (dt * dt);
  objective += subsystem_.elasticEnergy(local_q);

  for (const FEMSubsystem::ActiveConstraint& constraint :
       subsystem_.active_constraints_) {
    const double value = local_q[constraint.qOffset];
    const double diff = value - constraint.target;
    objective += 0.5 * constraint.stiffness * diff * diff;
  }
  return objective;
}

void FEMCPUBackend::updateInternalConstraints(double time, double dt)
{
  (void)dt;
  subsystem_.updateConstraintTargets(time);
}

void FEMCPUBackend::prepareLocalOperator(double dt)
{
  if (mass_diagonal_.size() != subsystem_.range_.scalarCount) {
    mass_diagonal_ = buildMassDiagonal();
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(subsystem_.range_.scalarCount + 144));
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    triplets.emplace_back(i, i, mass_diagonal_[i] / (dt * dt));
  }
  for (const FEMSubsystem::ActiveConstraint& constraint :
       subsystem_.active_constraints_) {
    triplets.emplace_back(
        constraint.qOffset, constraint.qOffset, constraint.stiffness);
  }
  subsystem_.assembleElasticHessian(subsystem_.gatherCurrentState(), triplets);

  local_matrix_.resize(subsystem_.range_.scalarCount,
                       subsystem_.range_.scalarCount);
  local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
  local_matrix_.makeCompressed();
}

void FEMCPUBackend::assembleLocalGradient(runtime::DofBuffer& g) const
{
  requireCPU(g, "FEMCPUBackend::assembleLocalGradient");
  ensureWritable(g.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);
  if (step_dt_ <= 0.0) {
    throw std::runtime_error("FEM CPU backend step has not begun");
  }

  const Eigen::VectorXd current = subsystem_.gatherCurrentState();
  Eigen::VectorXd local_gradient =
      (mass_diagonal_.array() * (current - inertial_target_).array()).matrix() /
      (step_dt_ * step_dt_);
  subsystem_.assembleElasticGradient(current, local_gradient);
  for (const FEMSubsystem::ActiveConstraint& constraint :
       subsystem_.active_constraints_) {
    local_gradient[constraint.qOffset] +=
        constraint.stiffness *
        (current[constraint.qOffset] - constraint.target);
  }
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    subsystem_.addToVector(g.cpu(), i, local_gradient[i]);
  }
}

void FEMCPUBackend::applyLocalMatrix(const runtime::DofBuffer& x,
                                     runtime::DofBuffer& y) const
{
  requireCPU(x, "FEMCPUBackend::applyLocalMatrix");
  requireCPU(y, "FEMCPUBackend::applyLocalMatrix");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend local operator is not prepared");
  }

  ensureWritable(y.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);
  const Eigen::VectorXd local_x = subsystem_.gatherLocalVector(x.cpu());
  const Eigen::VectorXd local_y = local_matrix_ * local_x;
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    subsystem_.addToVector(y.cpu(), i, local_y[i]);
  }
}

void FEMCPUBackend::solveLocalSystem(const runtime::DofBuffer& b,
                                     runtime::DofBuffer& x) const
{
  requireCPU(b, "FEMCPUBackend::solveLocalSystem");
  requireCPU(x, "FEMCPUBackend::solveLocalSystem");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend local operator is not prepared");
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(local_matrix_);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor FEM CPU backend local matrix");
  }
  const Eigen::VectorXd local_x =
      solver.solve(subsystem_.gatherLocalVector(b.cpu()));
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to solve FEM CPU backend local system");
  }

  ensureWritable(x.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    x.cpu()[subsystem_.vectorOffset(x.cpu(), i)] = local_x[i];
  }
}

void FEMCPUBackend::mapDirectionToGeometry(const runtime::DofBuffer& dq,
                                           runtime::GeometryBuffer& dx) const
{
  requireCPU(dq, "FEMCPUBackend::mapDirectionToGeometry");
  requireCPU(dx, "FEMCPUBackend::mapDirectionToGeometry");
  if (subsystem_.geometry_points_.size() != subsystem_.samples_.size()) {
    throw std::runtime_error("FEM geometry point mapping is stale");
  }

  for (int sample_index = 0;
       sample_index < static_cast<int>(subsystem_.samples_.size());
       ++sample_index) {
    const runtime::GeometryPointId point =
        subsystem_.geometry_points_[sample_index];
    if (point < 0 || point >= static_cast<int>(dx.cpu().size())) {
      throw std::invalid_argument(
          "FEM global geometry direction buffer is too small");
    }
    const int offset = subsystem_.samples_[sample_index].qOffset;
    dx.cpu()[static_cast<size_t>(point)] =
        glm::dvec3(dq.cpu()[subsystem_.vectorOffset(dq.cpu(), offset + 0)],
                   dq.cpu()[subsystem_.vectorOffset(dq.cpu(), offset + 1)],
                   dq.cpu()[subsystem_.vectorOffset(dq.cpu(), offset + 2)]);
  }
}

void FEMCPUBackend::scatterContactGradient(
    std::span<const runtime::GeometryPointId> points,
    const runtime::GeometryBuffer& pointGradient,
    runtime::DofBuffer& g) const
{
  requireCPU(pointGradient, "FEMCPUBackend::scatterContactGradient");
  requireCPU(g, "FEMCPUBackend::scatterContactGradient");
  if (points.size() != pointGradient.cpu().size()) {
    throw std::invalid_argument("contact point and gradient counts differ");
  }
  ensureWritable(g.cpu(),
                 subsystem_.range_.scalarOffset + subsystem_.range_.scalarCount);

  for (size_t i = 0; i < points.size(); ++i) {
    const auto it = std::find(subsystem_.geometry_points_.begin(),
                              subsystem_.geometry_points_.end(),
                              points[i]);
    if (it == subsystem_.geometry_points_.end()) {
      continue;
    }
    const int sample_index =
        static_cast<int>(it - subsystem_.geometry_points_.begin());
    const int offset = subsystem_.samples_[sample_index].qOffset;
    subsystem_.addToVector(g.cpu(), offset + 0, pointGradient.cpu()[i].x);
    subsystem_.addToVector(g.cpu(), offset + 1, pointGradient.cpu()[i].y);
    subsystem_.addToVector(g.cpu(), offset + 2, pointGradient.cpu()[i].z);
  }
}

void FEMCPUBackend::applyContactHessian(const runtime::DofBuffer& dq,
                                        const runtime::ContactTable& contacts,
                                        runtime::DofBuffer& y) const
{
  (void)dq;
  (void)contacts;
  (void)y;
}

Eigen::VectorXd FEMCPUBackend::buildMassDiagonal() const
{
  Eigen::VectorXd mass =
      Eigen::VectorXd::Zero(subsystem_.range_.scalarCount);
  for (int mesh_index = 0;
       mesh_index < static_cast<int>(subsystem_.meshes_.size());
       ++mesh_index) {
    const TetMeshDesc& mesh = subsystem_.meshes_[mesh_index];
    const double vertex_mass = std::max(1.0e-12, mesh.material.density);
    const int base = subsystem_.mesh_offsets_[mesh_index].q;
    for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
         ++vertex) {
      mass[base + 3 * vertex + 0] = vertex_mass;
      mass[base + 3 * vertex + 1] = vertex_mass;
      mass[base + 3 * vertex + 2] = vertex_mass;
    }
  }
  return mass;
}

}  // namespace ksk::engine::fem
