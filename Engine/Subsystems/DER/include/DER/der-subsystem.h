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
  void scatterContactGradient(
      std::span<const runtime::GeometryPointId> points,
      const runtime::GeometryBuffer& pointGradient,
      runtime::DofBuffer& g) const override;
  void applyContactHessian(const runtime::DofBuffer& dq,
                           const runtime::ContactTable& contacts,
                           runtime::DofBuffer& y) const override;
  void accept(runtime::SubsystemBackendVisitor& visitor) override;

  [[nodiscard]] const std::vector<Rod>& rods() const noexcept { return rods_; }
  [[nodiscard]] std::vector<Rod>& rods() noexcept { return rods_; }
  [[nodiscard]] const std::vector<DERRodOffset>& rodOffsets() const noexcept;
  [[nodiscard]] const std::vector<DERGeometrySample>& geometrySamples() const noexcept;
  [[nodiscard]] const std::vector<runtime::GeometryPointId>& geometryPointIds() const noexcept;
  [[nodiscard]] const RodEvaluation& cachedEvaluation(int rod) const;
  [[nodiscard]] int localScalarCount() const noexcept;
  [[nodiscard]] glm::dvec3 gravity() const noexcept { return gravity_; }

 private:
  void updateSceneConstraints(double time);
  void rebuildSamples();
  [[nodiscard]] bool usesGPU(const runtime::DofBuffer& buffer) const noexcept;
  [[nodiscard]] bool usesGPU(const runtime::GeometryBuffer& buffer) const noexcept;

  runtime::SubsystemId id_;
  runtime::DofRange range_;
  glm::dvec3 gravity_;
  std::vector<Rod> rods_;
  std::vector<DERConstraintBinding> constraints_;
  std::vector<DERRodOffset> rod_offsets_;
  std::vector<DERGeometrySample> samples_;
  std::vector<runtime::GeometryPointId> geometry_points_;
  std::unique_ptr<DERCPUBackend> cpu_backend_;
  std::unique_ptr<DERGpuBackend> gpu_backend_;
};

}  // namespace ksk::der
