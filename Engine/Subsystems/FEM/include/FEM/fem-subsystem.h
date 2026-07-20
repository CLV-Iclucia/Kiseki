#pragma once

#include <FEM/fem-scene.h>

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/geometry-transfer.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/subsystem.h>

#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <glm/glm.hpp>

namespace ksk::engine::fem {

class FEMCPUBackend;

struct FEMVertexSample {
  int mesh = -1;
  int vertex = -1;
  int qOffset = -1;
};

struct FEMMeshOffset {
  int q = 0;
  int samples = 0;
};

struct FEMConstraintBinding {
  int mesh = -1;
  runtime::SceneConstraintDesc constraint;
};

struct FEMMeshRuntimeRef {
  runtime::GeometryRange points;
  runtime::GeometryRange surfaceEdges;
  runtime::GeometryRange surfaceTriangles;
  runtime::GeometryRange tets;
  runtime::GeometryTransferMap transfer;
  TetMaterial material;
};

class FEMSubsystem final : public runtime::Subsystem {
 public:
  FEMSubsystem(runtime::SubsystemId id,
               std::vector<TetMeshDesc> meshes,
               std::vector<FEMConstraintBinding> constraints = {},
               int scalarOffset = 0,
               glm::dvec3 gravity = glm::dvec3(0.0));
  ~FEMSubsystem() override;

  [[nodiscard]] runtime::SubsystemId id() const noexcept override;
  [[nodiscard]] runtime::DofRange dofRange() const noexcept override;

  void declareGeometry(runtime::GlobalGeometryManager& geometry) override;
  void writeState(runtime::DofView q,
                  runtime::DofView qdot) const override;
  void readState(runtime::ConstDofView q,
                 runtime::ConstDofView qdot) override;
  void beginStep(runtime::ConstDofView q,
                 runtime::ConstDofView qdot,
                 double dt) override;
  void acceptStep(runtime::ConstDofView q,
                  runtime::DofView qdot,
                  double dt) override;
  [[nodiscard]] double evaluateObjective(runtime::ConstDofView q,
                                         runtime::ConstDofView qdot,
                                         double dt) override;

  void updateInternalConstraints(double time, double dt) override;
  void prepareLocalOperator(double dt) override;
  void assembleLocalGradient(runtime::DofView g) const override;
  void applyLocalMatrix(runtime::ConstDofView x,
                        runtime::DofView y) const override;
  void solveLocalSystem(runtime::ConstDofView b,
                        runtime::DofView x) const override;

  void updateGeometry(runtime::GlobalGeometryManager& geometry) const override;
  void mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                              runtime::GeometryView globalDx) const override;
  void applyInternalContacts(runtime::ContactStencils contacts) override;
  void scatterContactGradient(
      std::span<const runtime::PointIdx> points,
      runtime::ConstGeometryView pointGradient,
      runtime::DofView g) const override;
  void applyContactGeometryHessianProduct(
      std::span<const runtime::PointIdx> gradientPoints,
      runtime::ConstGeometryView pointGradient,
      std::span<const runtime::PointIdx> productPoints,
      runtime::ConstGeometryView pointHessianProduct,
      runtime::ConstDofView localDq,
      runtime::DofView localY) const override;
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const override;
  void visit(runtime::CpuSubsystemBackend& backend) override;
  void visit(runtime::GpuSubsystemBackend& backend) override;

  [[nodiscard]] const std::vector<TetMeshDesc>& meshes() const noexcept
  {
    return meshes_;
  }
  [[nodiscard]] int localScalarCount() const noexcept;
  [[nodiscard]] const std::vector<FEMVertexSample>& samples() const noexcept
  {
    return samples_;
  }
  [[nodiscard]] const std::vector<runtime::PointIdx>& geometryPointIds()
      const noexcept
  {
    return geometry_points_;
  }
  [[nodiscard]] const std::vector<FEMMeshRuntimeRef>& runtimeMeshes()
      const noexcept
  {
    return runtime_meshes_;
  }

 private:
  friend class FEMCPUBackend;

  struct ActiveConstraint {
    int qOffset = -1;
    double stiffness = 0.0;
    double target = 0.0;
  };

  void rebuildSamples();
  void updateConstraintTargets(double time);
  [[nodiscard]] Eigen::VectorXd gatherCurrentState() const;
  [[nodiscard]] double elasticEnergy(const Eigen::VectorXd& localQ) const;
  void assembleElasticGradient(const Eigen::VectorXd& localQ,
                               Eigen::VectorXd& gradient) const;
  void assembleElasticHessian(
      const Eigen::VectorXd& localQ,
      std::vector<Eigen::Triplet<double>>& triplets) const;
  [[nodiscard]] glm::dvec3 vertexPosition(int mesh, int vertex) const;
  [[nodiscard]] glm::dvec3 restVertexPosition(int mesh, int vertex) const;
  void setVertexPosition(int mesh, int vertex, const glm::dvec3& position);
  [[nodiscard]] glm::dvec3 vertexVelocity(int mesh, int vertex) const;
  void setVertexVelocity(int mesh, int vertex, const glm::dvec3& velocity);
  [[nodiscard]] int constraintLocalOffset(
      const FEMConstraintBinding& binding) const;
  runtime::SubsystemId id_;
  runtime::DofRange range_;
  glm::dvec3 gravity_;
  std::vector<TetMeshDesc> meshes_;
  std::vector<FEMConstraintBinding> constraints_;
  std::vector<ActiveConstraint> active_constraints_;
  std::vector<FEMMeshOffset> mesh_offsets_;
  std::vector<FEMMeshRuntimeRef> runtime_meshes_;
  std::vector<FEMVertexSample> samples_;
  std::vector<runtime::PointIdx> geometry_points_;
  runtime::ContactStencils internal_contacts_;
  std::unique_ptr<FEMCPUBackend> cpu_backend_;
};

}  // namespace ksk::engine::fem
