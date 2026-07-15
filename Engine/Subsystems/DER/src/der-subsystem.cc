#include <DER/der-subsystem.h>

#include <DER/cpu/der-cpu-backend.h>
#include <DER/gpu/der-gpu-backend.h>

#include <stdexcept>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::der {
namespace {

int rodScalarCount(const Rod& rod)
{
  return static_cast<int>(4 * rod.state().size());
}

[[nodiscard]] std::optional<RodConstraintProperty> rodConstraintProperty(
    std::string_view property)
{
  if (property == "x") {
    return RodConstraintProperty::X;
  }
  if (property == "y") {
    return RodConstraintProperty::Y;
  }
  if (property == "z") {
    return RodConstraintProperty::Z;
  }
  if (property == "twist") {
    return RodConstraintProperty::Twist;
  }
  return std::nullopt;
}

}  // namespace

DERSubsystem::DERSubsystem(runtime::SubsystemId id,
                           std::vector<Rod> rods,
                           std::vector<DERConstraintBinding> constraints,
                           int scalarOffset,
                           glm::dvec3 gravity)
    : id_(id),
      gravity_(gravity),
      rods_(std::move(rods)),
      constraints_(std::move(constraints))
{
  int scalar_count = 0;
  rod_offsets_.reserve(rods_.size());
  int sample_count = 0;
  for (const Rod& rod : rods_) {
    rod_offsets_.push_back(DERRodOffset{
        .q = scalar_count,
        .samples = sample_count,
    });
    scalar_count += rodScalarCount(rod);
    sample_count += static_cast<int>(rod.state().size());
  }

  range_ = runtime::DofRange{
      .subsystem = id_,
      .scalarOffset = scalarOffset,
      .scalarCount = scalar_count,
      .blockSize = 0,
  };
  rebuildSamples();
  updateSceneConstraints(0.0);
  cpu_backend_ = std::make_unique<DERCPUBackend>(*this);
  gpu_backend_ = std::make_unique<DERGpuBackend>(*this);
}

DERSubsystem::~DERSubsystem() = default;

runtime::SubsystemId DERSubsystem::id() const noexcept
{
  return id_;
}

runtime::DofRange DERSubsystem::dofRange() const noexcept
{
  return range_;
}

void DERSubsystem::declareGeometry(runtime::GlobalGeometryManager& geometry)
{
  geometry_points_.clear();
  geometry_points_.reserve(samples_.size());

  for (int rod_index = 0; rod_index < static_cast<int>(rods_.size());
       ++rod_index) {
    const Rod& rod = rods_[rod_index];
    const int sample_base = rod_offsets_[rod_index].samples;
    for (int vertex = 0; vertex < static_cast<int>(rod.state().size());
         ++vertex) {
      const int sample = sample_base + vertex;
      geometry_points_.push_back(
          geometry.appendPoint(id_, sample, rod.state().position(vertex)));
    }

    for (int edge = 0; edge + 1 < static_cast<int>(rod.state().size());
         ++edge) {
      const runtime::GeometryPointId p0 = geometry_points_[sample_base + edge];
      const runtime::GeometryPointId p1 =
          geometry_points_[sample_base + edge + 1];
      geometry.appendEdge(p0, p1, rod.material().radius);
    }
  }
}

void DERSubsystem::writeState(runtime::DofBuffer& q,
                              runtime::DofBuffer& qdot) const
{
  if (usesGPU(q) || usesGPU(qdot)) {
    gpu_backend_->writeState(q, qdot);
    return;
  }
  cpu_backend_->writeState(q, qdot);
}

void DERSubsystem::readState(runtime::DofBuffer& q,
                             runtime::DofBuffer& qdot)
{
  if (usesGPU(q) || usesGPU(qdot)) {
    gpu_backend_->readState(q, qdot);
    return;
  }
  cpu_backend_->readState(q, qdot);
}

void DERSubsystem::beginStep(const runtime::DofBuffer& q,
                             const runtime::DofBuffer& qdot,
                             double dt)
{
  if (usesGPU(q) || usesGPU(qdot)) {
    gpu_backend_->beginStep(q, qdot, dt);
    return;
  }
  cpu_backend_->beginStep(q, qdot, dt);
}

void DERSubsystem::acceptStep(const runtime::DofBuffer& q,
                              runtime::DofBuffer& qdot,
                              double dt)
{
  if (usesGPU(q) || usesGPU(qdot)) {
    gpu_backend_->acceptStep(q, qdot, dt);
    return;
  }
  cpu_backend_->acceptStep(q, qdot, dt);
}

double DERSubsystem::evaluateObjective(const runtime::DofBuffer& q,
                                       const runtime::DofBuffer& qdot,
                                       double dt)
{
  if (usesGPU(q) || usesGPU(qdot)) {
    return gpu_backend_->evaluateObjective(q, qdot, dt);
  }
  return cpu_backend_->evaluateObjective(q, qdot, dt);
}

void DERSubsystem::updateInternalConstraints(double time, double dt)
{
  updateSceneConstraints(time);
  cpu_backend_->updateInternalConstraints(time, dt);
  gpu_backend_->updateInternalConstraints(time, dt);
}

void DERSubsystem::prepareLocalOperator(double dt)
{
  cpu_backend_->prepareLocalOperator(dt);
}

void DERSubsystem::assembleLocalGradient(runtime::DofBuffer& g) const
{
  if (usesGPU(g)) {
    gpu_backend_->assembleLocalGradient(g);
    return;
  }
  cpu_backend_->assembleLocalGradient(g);
}

void DERSubsystem::applyLocalMatrix(const runtime::DofBuffer& x,
                                    runtime::DofBuffer& y) const
{
  if (usesGPU(x) || usesGPU(y)) {
    gpu_backend_->applyLocalMatrix(x, y);
    return;
  }
  cpu_backend_->applyLocalMatrix(x, y);
}

void DERSubsystem::solveLocalSystem(const runtime::DofBuffer& b,
                                    runtime::DofBuffer& x) const
{
  if (usesGPU(b) || usesGPU(x)) {
    gpu_backend_->solveLocalSystem(b, x);
    return;
  }
  cpu_backend_->solveLocalSystem(b, x);
}

void DERSubsystem::updateGeometry(runtime::GlobalGeometryManager& geometry) const
{
  for (int sample = 0; sample < static_cast<int>(samples_.size()); ++sample) {
    const runtime::GeometryPointId point = geometry_points_.at(sample);
    if (!geometry.contains(point)) {
      throw std::runtime_error("DERSubsystem geometry point id is stale");
    }
    const DERGeometrySample& mapping = samples_[sample];
    geometry.setPointPosition(
        point,
        rods_[mapping.rod].state().position(mapping.vertex));
  }
}

void DERSubsystem::mapDirectionToGeometry(const runtime::DofBuffer& dq,
                                          runtime::GeometryBuffer& dx) const
{
  if (usesGPU(dq) || usesGPU(dx)) {
    gpu_backend_->mapDirectionToGeometry(dq, dx);
    return;
  }
  cpu_backend_->mapDirectionToGeometry(dq, dx);
}

void DERSubsystem::setInternalContacts(runtime::ContactTable contacts)
{
  // TODO: assemble DER-local contact energy/gradient/Hessian from these
  // candidates. The routing path is in place, but DER contact physics is not.
  internal_contacts_ = std::move(contacts);
}

void DERSubsystem::scatterContactGradient(
    std::span<const runtime::GeometryPointId> points,
    const runtime::GeometryBuffer& pointGradient,
    runtime::DofBuffer& g) const
{
  if (usesGPU(pointGradient) || usesGPU(g)) {
    gpu_backend_->scatterContactGradient(points, pointGradient, g);
    return;
  }
  cpu_backend_->scatterContactGradient(points, pointGradient, g);
}

void DERSubsystem::applyContactHessian(const runtime::DofBuffer& dq,
                                       const runtime::ContactTable& contacts,
                                       runtime::DofBuffer& y) const
{
  if (usesGPU(dq) || usesGPU(y)) {
    gpu_backend_->applyContactHessian(dq, contacts, y);
    return;
  }
  cpu_backend_->applyContactHessian(dq, contacts, y);
}

void DERSubsystem::visit(runtime::CpuSubsystemBackend& backend)
{
  (void)backend;
}

void DERSubsystem::visit(runtime::GpuSubsystemBackend& backend)
{
  (void)backend;
}

const std::vector<DERGeometrySample>& DERSubsystem::geometrySamples() const noexcept
{
  return samples_;
}

const std::vector<DERRodOffset>& DERSubsystem::rodOffsets() const noexcept
{
  return rod_offsets_;
}

const std::vector<runtime::GeometryPointId>& DERSubsystem::geometryPointIds() const noexcept
{
  return geometry_points_;
}

const RodEvaluation& DERSubsystem::cachedEvaluation(int rod) const
{
  return cpu_backend_->cachedEvaluation(rod);
}

int DERSubsystem::localScalarCount() const noexcept
{
  return range_.scalarCount;
}

void DERSubsystem::updateSceneConstraints(double time)
{
  for (Rod& rod : rods_) {
    rod.clearConstraints();
  }

  for (const DERConstraintBinding& binding : constraints_) {
    if (binding.rod < 0 || binding.rod >= static_cast<int>(rods_.size())) {
      throw std::runtime_error("DER constraint binding references an invalid rod");
    }

    const runtime::SceneConstraintDesc& scene_constraint =
        binding.constraint;
    Rod& rod = rods_[binding.rod];
    const std::optional<RodConstraintProperty> property =
        rodConstraintProperty(scene_constraint.property);
    if (!property) {
      throw std::runtime_error("DER constraint property is not supported: " +
                               scene_constraint.property);
    }

    if (!scene_constraint.target) {
      throw std::runtime_error("DER rod constraint requires a scalar target");
    }
    const double target = scene_constraint.target(time);
    rod.addConstraint(RodConstraint{
        .property = *property,
        .sample = static_cast<size_t>(scene_constraint.sample),
        .target = target,
        .stiffness = scene_constraint.stiffness,
    });
  }
}

void DERSubsystem::rebuildSamples()
{
  samples_.clear();
  samples_.reserve(static_cast<size_t>(range_.scalarCount / 4));
  for (int rod_index = 0; rod_index < static_cast<int>(rods_.size());
       ++rod_index) {
    const int base = rod_offsets_[rod_index].q;
    for (int vertex = 0; vertex < static_cast<int>(rods_[rod_index].state().size());
         ++vertex) {
      samples_.push_back(DERGeometrySample{
          .rod = rod_index,
          .vertex = vertex,
          .qOffset = base + 4 * vertex,
      });
    }
  }
}

bool DERSubsystem::usesGPU(const runtime::DofBuffer& buffer) const noexcept
{
  return buffer.isGPU();
}

bool DERSubsystem::usesGPU(const runtime::GeometryBuffer& buffer) const noexcept
{
  return buffer.isGPU();
}

}  // namespace ksk::der
