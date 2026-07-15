#pragma once

#include <FEM/fem-scene.h>

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
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
  void writeState(runtime::DofBuffer& q,
                  runtime::DofBuffer& qdot) const override;
  void readState(runtime::DofBuffer& q,
                 runtime::DofBuffer& qdot) override;
  void beginStep(const runtime::DofBuffer& q,
                 const runtime::DofBuffer& qdot,
                 double dt) override;
  void acceptStep(const runtime::DofBuffer& q,
                  runtime::DofBuffer& qdot,
                  double dt) override;
  [[nodiscard]] double evaluateObjective(const runtime::DofBuffer& q,
                                         const runtime::DofBuffer& qdot,
                                         double dt) override;

  void updateInternalConstraints(double time, double dt) override;
  void prepareLocalOperator(double dt) override;
  void assembleLocalGradient(runtime::DofBuffer& g) const override;
  void applyLocalMatrix(const runtime::DofBuffer& x,
                        runtime::DofBuffer& y) const override;
  void solveLocalSystem(const runtime::DofBuffer& b,
                        runtime::DofBuffer& x) const override;

  void updateGeometry(runtime::GlobalGeometryManager& geometry) const override;
  void mapDirectionToGeometry(const runtime::DofBuffer& dq,
                              runtime::GeometryBuffer& dx) const override;
  void setInternalContacts(runtime::ContactTable contacts) override;
  void scatterContactGradient(
      std::span<const runtime::GeometryPointId> points,
      const runtime::GeometryBuffer& pointGradient,
      runtime::DofBuffer& g) const override;
  void applyContactHessian(const runtime::DofBuffer& dq,
                           const runtime::ContactTable& contacts,
                           runtime::DofBuffer& y) const override;
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
  [[nodiscard]] const std::vector<runtime::GeometryPointId>& geometryPointIds()
      const noexcept
  {
    return geometry_points_;
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
  [[nodiscard]] int vectorOffset(const Eigen::VectorXd& values,
                                 int localOffset) const;
  [[nodiscard]] Eigen::VectorXd gatherLocalVector(
      const Eigen::VectorXd& values) const;
  [[nodiscard]] glm::dvec3 vertexPosition(int mesh, int vertex) const;
  void setVertexPosition(int mesh, int vertex, const glm::dvec3& position);
  [[nodiscard]] glm::dvec3 vertexVelocity(int mesh, int vertex) const;
  void setVertexVelocity(int mesh, int vertex, const glm::dvec3& velocity);
  [[nodiscard]] int constraintLocalOffset(
      const FEMConstraintBinding& binding) const;
  void addToVector(Eigen::VectorXd& values,
                   int localOffset,
                   double value) const;

  runtime::SubsystemId id_;
  runtime::DofRange range_;
  glm::dvec3 gravity_;
  std::vector<TetMeshDesc> meshes_;
  std::vector<FEMConstraintBinding> constraints_;
  std::vector<ActiveConstraint> active_constraints_;
  std::vector<FEMMeshOffset> mesh_offsets_;
  std::vector<FEMVertexSample> samples_;
  std::vector<runtime::GeometryPointId> geometry_points_;
  runtime::ContactTable internal_contacts_;
  std::unique_ptr<FEMCPUBackend> cpu_backend_;
};

}  // namespace ksk::engine::fem
