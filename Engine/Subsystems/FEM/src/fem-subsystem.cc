#include <FEM/fem-subsystem.h>

#include <FEM/cpu/fem-cpu-backend.h>

#include <Deform/deformation-gradient.h>
#include <Deform/strain-energy-density.h>
#include <Maths/tensor.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::engine::fem {
namespace {

int vertexScalarCount(const TetMeshDesc& mesh)
{
  return 3 * static_cast<int>(mesh.vertices.size());
}

glm::dvec3 initialPosition(const TetMeshDesc& mesh, int vertex)
{
  if (!mesh.initialPositions.empty()) {
    return mesh.initialPositions.at(static_cast<size_t>(vertex));
  }
  return mesh.vertices.at(static_cast<size_t>(vertex));
}

glm::dvec3 initialVelocity(const TetMeshDesc& mesh, int vertex)
{
  if (!mesh.initialVelocities.empty()) {
    return mesh.initialVelocities.at(static_cast<size_t>(vertex));
  }
  return glm::dvec3(0.0);
}

int propertyLane(const std::string& property)
{
  if (property == "x") {
    return 0;
  }
  if (property == "y") {
    return 1;
  }
  if (property == "z") {
    return 2;
  }
  throw std::runtime_error("FEM constraint property is not supported: " +
                           property);
}

deform::StableNeoHookean<double> createEnergy(const TetMaterial& material)
{
  return deform::StableNeoHookean<double>(
      deform::ElasticityParameters<double>{
          .E = material.youngsModulus,
          .nu = material.poissonRatio,
      });
}

Eigen::Matrix3d tetEdges(const TetMeshDesc& mesh,
                         const Eigen::VectorXd& localQ,
                         int meshBase,
                         const std::array<int, 4>& tet)
{
  Eigen::Matrix3d edges;
  const int root = meshBase + 3 * tet[0];
  for (int column = 0; column < 3; ++column) {
    const int vertex = meshBase + 3 * tet[column + 1];
    edges.col(column) =
        localQ.segment<3>(vertex) - localQ.segment<3>(root);
  }
  return edges;
}

Eigen::Matrix3d restTetEdges(const TetMeshDesc& mesh,
                             const std::array<int, 4>& tet)
{
  Eigen::Matrix3d edges;
  const glm::dvec3 root = mesh.vertices.at(static_cast<size_t>(tet[0]));
  for (int column = 0; column < 3; ++column) {
    const glm::dvec3 vertex =
        mesh.vertices.at(static_cast<size_t>(tet[column + 1]));
    const glm::dvec3 edge = vertex - root;
    edges.col(column) = Eigen::Vector3d(edge.x, edge.y, edge.z);
  }
  return edges;
}

}  // namespace

FEMSubsystem::FEMSubsystem(runtime::SubsystemId id,
                           std::vector<TetMeshDesc> meshes,
                           std::vector<FEMConstraintBinding> constraints,
                           int scalarOffset,
                           glm::dvec3 gravity)
    : id_(id)
    , gravity_(gravity)
    , meshes_(std::move(meshes))
    , constraints_(std::move(constraints))
{
  int scalar_count = 0;
  int sample_count = 0;
  mesh_offsets_.reserve(meshes_.size());
  for (const TetMeshDesc& mesh : meshes_) {
    mesh_offsets_.push_back(FEMMeshOffset{
        .q = scalar_count,
        .samples = sample_count,
    });
    scalar_count += vertexScalarCount(mesh);
    sample_count += static_cast<int>(mesh.vertices.size());
  }

  range_ = runtime::DofRange{
      .subsystem = id_,
      .scalarOffset = scalarOffset,
      .scalarCount = scalar_count,
      .blockSize = 3,
  };
  rebuildSamples();
  updateConstraintTargets(0.0);
  cpu_backend_ = std::make_unique<FEMCPUBackend>(*this);
}

FEMSubsystem::~FEMSubsystem() = default;

runtime::SubsystemId FEMSubsystem::id() const noexcept
{
  return id_;
}

runtime::DofRange FEMSubsystem::dofRange() const noexcept
{
  return range_;
}

void FEMSubsystem::declareGeometry(runtime::GlobalGeometryManager& geometry)
{
  geometry_points_.clear();
  geometry_points_.reserve(samples_.size());

  for (int mesh_index = 0; mesh_index < meshes_.size(); mesh_index++) {
    const TetMeshDesc& mesh = meshes_[mesh_index];
    const int sample_base = mesh_offsets_[mesh_index].samples;
    for (int vertex = 0; vertex < mesh.vertices.size(); vertex++) {
      geometry_points_.push_back(geometry.addPoint(
          id_,
          sample_base + vertex,
          vertexPosition(mesh_index, vertex),
          restVertexPosition(mesh_index, vertex)));
    }

    for (const auto& edge : mesh.surfaceEdges) {
      geometry.addEdge(geometry_points_.at(sample_base + edge[0]),
                          geometry_points_.at(sample_base + edge[1]));
    }
    for (const auto& triangle : mesh.surfaceTriangles) {
      geometry.addTriangle(geometry_points_.at(sample_base + triangle[0]),
                              geometry_points_.at(sample_base + triangle[1]),
                              geometry_points_.at(sample_base + triangle[2]));
    }
  }
}

void FEMSubsystem::writeState(runtime::DofView q,
                              runtime::DofView qdot) const
{
  cpu_backend_->writeState(q, qdot);
}

void FEMSubsystem::readState(runtime::ConstDofView q,
                             runtime::ConstDofView qdot)
{
  cpu_backend_->readState(q, qdot);
}

void FEMSubsystem::beginStep(runtime::ConstDofView q,
                             runtime::ConstDofView qdot,
                             double dt)
{
  cpu_backend_->beginStep(q, qdot, dt);
}

void FEMSubsystem::acceptStep(runtime::ConstDofView q,
                              runtime::DofView qdot,
                              double dt)
{
  cpu_backend_->acceptStep(q, qdot, dt);
}

double FEMSubsystem::evaluateObjective(runtime::ConstDofView q,
                                       runtime::ConstDofView qdot,
                                       double dt)
{
  return cpu_backend_->evaluateObjective(q, qdot, dt);
}

void FEMSubsystem::updateInternalConstraints(double time, double dt)
{
  cpu_backend_->updateInternalConstraints(time, dt);
}

void FEMSubsystem::prepareLocalOperator(double dt)
{
  cpu_backend_->prepareLocalOperator(dt);
}

void FEMSubsystem::assembleLocalGradient(runtime::DofView g) const
{
  cpu_backend_->assembleLocalGradient(g);
}

void FEMSubsystem::applyLocalMatrix(runtime::ConstDofView x,
                                    runtime::DofView y) const
{
  cpu_backend_->applyLocalMatrix(x, y);
}

void FEMSubsystem::solveLocalSystem(runtime::ConstDofView b,
                                    runtime::DofView x) const
{
  cpu_backend_->solveLocalSystem(b, x);
}

void FEMSubsystem::updateGeometry(runtime::GlobalGeometryManager& geometry) const
{
  for (int sample_index = 0; sample_index < static_cast<int>(samples_.size());
       ++sample_index) {
    const runtime::PointIdx point = geometry_points_.at(sample_index);
    if (!geometry.contains(point)) {
      throw std::runtime_error("FEM geometry point id is stale");
    }
    const FEMVertexSample& sample = samples_[sample_index];
    geometry.setPointPosition(point, vertexPosition(sample.mesh, sample.vertex));
    geometry.setPointRestPosition(
        point, restVertexPosition(sample.mesh, sample.vertex));
  }
}

void FEMSubsystem::mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                          runtime::GeometryView globalDx) const
{
  cpu_backend_->mapLocalDirectionToGeometry(localDq, globalDx);
}

void FEMSubsystem::applyInternalContacts(runtime::ContactStencils contacts)
{
  // The CPU backend consumes these routed internal contacts when assembling
  // contact energy, gradient, and Hessian products.
  internal_contacts_ = std::move(contacts);
}

void FEMSubsystem::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient,
    runtime::DofView g) const
{
  cpu_backend_->scatterContactGradient(points, pointGradient, g);
}

void FEMSubsystem::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> gradientPoints,
    runtime::ConstGeometryView pointGradient,
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::ConstDofView localDq,
    runtime::DofView localY) const
{
  cpu_backend_->applyContactGeometryHessianProduct(
      gradientPoints,
      pointGradient,
      productPoints,
      pointHessianProduct,
      localDq,
      localY);
}

void FEMSubsystem::applyInternalContactHessian(runtime::ConstDofView localDq,
                                               runtime::DofView localY) const
{
  cpu_backend_->applyInternalContactHessian(localDq, localY);
}

void FEMSubsystem::visit(runtime::CpuSubsystemBackend& backend)
{
}

void FEMSubsystem::visit(runtime::GpuSubsystemBackend& backend)
{
}

int FEMSubsystem::localScalarCount() const noexcept
{
  return range_.scalarCount;
}

void FEMSubsystem::rebuildSamples()
{
  samples_.clear();
  samples_.reserve(static_cast<size_t>(range_.scalarCount / 3));
  for (int mesh_index = 0; mesh_index < static_cast<int>(meshes_.size());
       ++mesh_index) {
    const int base = mesh_offsets_[mesh_index].q;
    for (int vertex = 0; vertex < static_cast<int>(meshes_[mesh_index].vertices.size());
         ++vertex) {
      samples_.push_back(FEMVertexSample{
          .mesh = mesh_index,
          .vertex = vertex,
          .qOffset = base + 3 * vertex,
      });
    }
  }
}

void FEMSubsystem::updateConstraintTargets(double time)
{
  active_constraints_.clear();
  active_constraints_.reserve(constraints_.size());
  for (const FEMConstraintBinding& binding : constraints_) {
    const int offset = constraintLocalOffset(binding);
    if (!binding.constraint.target) {
      throw std::runtime_error("FEM constraint requires a scalar target");
    }
    active_constraints_.push_back(ActiveConstraint{
        .qOffset = offset,
        .stiffness = binding.constraint.stiffness,
        .target = binding.constraint.target(time),
    });
  }
}

Eigen::VectorXd FEMSubsystem::gatherCurrentState() const
{
  Eigen::VectorXd values(range_.scalarCount);
  for (const FEMVertexSample& sample : samples_) {
    const glm::dvec3 position = vertexPosition(sample.mesh, sample.vertex);
    values[sample.qOffset + 0] = position.x;
    values[sample.qOffset + 1] = position.y;
    values[sample.qOffset + 2] = position.z;
  }
  return values;
}

double FEMSubsystem::elasticEnergy(const Eigen::VectorXd& localQ) const
{
  double energy = 0.0;
  for (int mesh_index = 0; mesh_index < static_cast<int>(meshes_.size());
       ++mesh_index) {
    const TetMeshDesc& mesh = meshes_[mesh_index];
    const int mesh_base = mesh_offsets_[mesh_index].q;
    const auto model = createEnergy(mesh.material);
    for (const auto& tet : mesh.tets) {
      const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
      const double volume = local_X.determinant() / 6.0;
      if (volume <= 0.0) {
        throw std::runtime_error("FEM tet has non-positive rest volume");
      }
      deform::DeformationGradient<double, 3> dg(local_X);
      dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
      energy += model.computeEnergyDensity(dg) * volume;
    }
  }
  return energy;
}

void FEMSubsystem::assembleElasticGradient(const Eigen::VectorXd& localQ,
                                           Eigen::VectorXd& gradient) const
{
  for (int mesh_index = 0; mesh_index < static_cast<int>(meshes_.size());
       ++mesh_index) {
    const TetMeshDesc& mesh = meshes_[mesh_index];
    const int mesh_base = mesh_offsets_[mesh_index].q;
    const auto model = createEnergy(mesh.material);
    for (const auto& tet : mesh.tets) {
      const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
      const double volume = local_X.determinant() / 6.0;
      if (volume <= 0.0) {
        throw std::runtime_error("FEM tet has non-positive rest volume");
      }
      deform::DeformationGradient<double, 3> dg(local_X);
      dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
      const Eigen::Matrix<double, 12, 1> local_gradient =
          dg.gradient().transpose() *
          maths::vectorize(model.computeEnergyGradient(dg)) * volume;
      for (int vertex = 0; vertex < 4; ++vertex) {
        gradient.segment<3>(mesh_base + 3 * tet[vertex]) +=
            local_gradient.segment<3>(3 * vertex);
      }
    }
  }
}

void FEMSubsystem::assembleElasticHessian(
    const Eigen::VectorXd& localQ,
    std::vector<Eigen::Triplet<double>>& triplets) const
{
  for (int mesh_index = 0; mesh_index < static_cast<int>(meshes_.size());
       ++mesh_index) {
    const TetMeshDesc& mesh = meshes_[mesh_index];
    const int mesh_base = mesh_offsets_[mesh_index].q;
    const auto model = createEnergy(mesh.material);
    for (const auto& tet : mesh.tets) {
      const Eigen::Matrix3d local_X = restTetEdges(mesh, tet);
      const double volume = local_X.determinant() / 6.0;
      if (volume <= 0.0) {
        throw std::runtime_error("FEM tet has non-positive rest volume");
      }
      deform::DeformationGradient<double, 3> dg(local_X);
      dg.updateCurrentConfig(tetEdges(mesh, localQ, mesh_base, tet));
      const Eigen::Matrix<double, 12, 12> local_hessian =
          dg.gradient().transpose() * model.filteredEnergyHessian(dg) *
          dg.gradient() * volume;
      for (int row_vertex = 0; row_vertex < 4; ++row_vertex) {
        for (int col_vertex = 0; col_vertex < 4; ++col_vertex) {
          for (int row_lane = 0; row_lane < 3; ++row_lane) {
            for (int col_lane = 0; col_lane < 3; ++col_lane) {
              const int row = mesh_base + 3 * tet[row_vertex] + row_lane;
              const int col = mesh_base + 3 * tet[col_vertex] + col_lane;
              const double value =
                  local_hessian(3 * row_vertex + row_lane,
                                3 * col_vertex + col_lane);
              if (value != 0.0) {
                triplets.emplace_back(row, col, value);
              }
            }
          }
        }
      }
    }
  }
}

glm::dvec3 FEMSubsystem::vertexPosition(int mesh, int vertex) const
{
  return initialPosition(meshes_.at(mesh), vertex);
}

glm::dvec3 FEMSubsystem::restVertexPosition(int mesh, int vertex) const
{
  return meshes_.at(mesh).vertices.at(vertex);
}

void FEMSubsystem::setVertexPosition(int mesh,
                                     int vertex,
                                     const glm::dvec3& position)
{
  TetMeshDesc& desc = meshes_.at(mesh);
  if (desc.initialPositions.empty()) {
    desc.initialPositions = desc.vertices;
  }
  desc.initialPositions.at(vertex) = position;
}

glm::dvec3 FEMSubsystem::vertexVelocity(int mesh, int vertex) const
{
  return initialVelocity(meshes_.at(mesh), vertex);
}

void FEMSubsystem::setVertexVelocity(int mesh,
                                     int vertex,
                                     const glm::dvec3& velocity)
{
  TetMeshDesc& desc = meshes_.at(mesh);
  if (desc.initialVelocities.empty()) {
    desc.initialVelocities.resize(desc.vertices.size(), glm::dvec3(0.0));
  }
  desc.initialVelocities.at(vertex) = velocity;
}

int FEMSubsystem::constraintLocalOffset(
    const FEMConstraintBinding& binding) const
{
  const TetMeshDesc& mesh = meshes_[binding.mesh];
  const int lane = propertyLane(binding.constraint.property);
  return mesh_offsets_[binding.mesh].q + 3 * binding.constraint.sample + lane;
}

}  // namespace ksk::engine::fem
