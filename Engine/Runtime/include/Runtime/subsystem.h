#pragma once

#include <Runtime/contact-table.h>
#include <Runtime/buffers.h>
#include <Runtime/dof-layout.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/subsystem-backend.h>

#include <span>

namespace ksk::runtime {

class Subsystem {
 public:
  virtual ~Subsystem() = default;

  [[nodiscard]] virtual SubsystemId id() const noexcept = 0;
  [[nodiscard]] virtual DofRange dofRange() const noexcept = 0;

  virtual void declareGeometry(GlobalGeometryManager& geometry) = 0;
  virtual void writeState(DofView q, DofView qdot) const = 0;
  virtual void readState(ConstDofView q, ConstDofView qdot) = 0;
  virtual void beginStep(ConstDofView q,
                         ConstDofView qdot,
                         double dt) = 0;
  virtual void acceptStep(ConstDofView q,
                          DofView qdot,
                          double dt) = 0;
  [[nodiscard]] virtual double evaluateObjective(ConstDofView q,
                                                 ConstDofView qdot,
                                                 double dt) = 0;

  virtual void updateInternalConstraints(double time, double dt) = 0;
  virtual void prepareLocalOperator(double dt) = 0;
  virtual void assembleLocalGradient(DofView g) const = 0;
  virtual void applyLocalMatrix(ConstDofView x, DofView y) const = 0;
  virtual void solveLocalSystem(ConstDofView b, DofView x) const = 0;

  virtual void updateGeometry(GlobalGeometryManager& geometry) const = 0;
  virtual void mapLocalDirectionToGeometry(ConstDofView localDq,
                                      GeometryView globalDx) const = 0;
  virtual void applyInternalContacts(ContactStencils contacts) = 0;
  virtual void scatterContactGradient(
      std::span<const PointIdx> points,
      ConstGeometryView pointGradient,
      DofView g) const = 0;
  virtual void applyInternalContactHessian(ConstDofView localDq,
                                           DofView localY) const = 0;
  virtual void visit(CpuSubsystemBackend& backend) = 0;
  virtual void visit(GpuSubsystemBackend& backend) = 0;
};

}  // namespace ksk::runtime
