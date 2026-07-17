#pragma once

#include <DER/rod.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/subsystem.h>

#include <glm/glm.hpp>

#include <memory>
#include <span>
#include <vector>

namespace ksk::der {

class DERCPUBackend;
class DERGpuBackend;

struct DERGeometrySample {
  int rod = -1;
  int vertex = -1;
  int qOffset = -1;
};

struct DERRodOffset {
  int q = 0;
  int samples = 0;
};

struct DERConstraintBinding {
  int rod = -1;
  runtime::SceneConstraintDesc constraint;
};

class DERSubsystem final : public runtime::Subsystem {
 public:
  DERSubsystem(runtime::SubsystemId id,
               std::vector<Rod> rods,
               std::vector<DERConstraintBinding> constraints = {},
               int scalarOffset = 0,
               glm::dvec3 gravity = glm::dvec3(0.0));
  ~DERSubsystem() override;

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
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const override;
  void visit(runtime::CpuSubsystemBackend& backend) override;
  void visit(runtime::GpuSubsystemBackend& backend) override;

  [[nodiscard]] const std::vector<Rod>& rods() const noexcept { return rods_; }
  [[nodiscard]] std::vector<Rod>& rods() noexcept { return rods_; }
  [[nodiscard]] const std::vector<DERRodOffset>& rodOffsets() const noexcept;
  [[nodiscard]] const std::vector<DERGeometrySample>& geometrySamples() const noexcept;
  [[nodiscard]] const std::vector<runtime::PointIdx>& geometryPointIds() const noexcept;
  [[nodiscard]] const RodEvaluation& cachedEvaluation(int rod) const;
  [[nodiscard]] int localScalarCount() const noexcept;
  [[nodiscard]] glm::dvec3 gravity() const noexcept { return gravity_; }

 private:
  friend class DERCPUBackend;

  void updateSceneConstraints(double time);
  void rebuildSamples();
  [[nodiscard]] bool usesGPU(runtime::ConstDofView view) const noexcept;
  [[nodiscard]] bool usesGPU(runtime::DofView view) const noexcept;
  [[nodiscard]] bool usesGPU(runtime::ConstGeometryView view) const noexcept;
  [[nodiscard]] bool usesGPU(runtime::GeometryView view) const noexcept;

  runtime::SubsystemId id_;
  runtime::DofRange range_;
  glm::dvec3 gravity_;
  std::vector<Rod> rods_;
  std::vector<DERConstraintBinding> constraints_;
  std::vector<DERRodOffset> rod_offsets_;
  std::vector<DERGeometrySample> samples_;
  std::vector<runtime::PointIdx> geometry_points_;
  runtime::ContactStencils internal_contacts_;
  std::unique_ptr<DERCPUBackend> cpu_backend_;
  std::unique_ptr<DERGpuBackend> gpu_backend_;
};

}  // namespace ksk::der
