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
  virtual void writeState(DofBuffer& q, DofBuffer& qdot) const = 0;
  virtual void readState(DofBuffer& q, DofBuffer& qdot) = 0;
  virtual void beginStep(const DofBuffer& q,
                         const DofBuffer& qdot,
                         double dt) = 0;
  virtual void acceptStep(const DofBuffer& q,
                          DofBuffer& qdot,
                          double dt) = 0;
  [[nodiscard]] virtual double evaluateObjective(const DofBuffer& q,
                                                 const DofBuffer& qdot,
                                                 double dt) = 0;

  virtual void updateInternalConstraints(double time, double dt) = 0;
  virtual void prepareLocalOperator(double dt) = 0;
  virtual void assembleLocalGradient(DofBuffer& g) const = 0;
  virtual void applyLocalMatrix(const DofBuffer& x, DofBuffer& y) const = 0;
  virtual void solveLocalSystem(const DofBuffer& b, DofBuffer& x) const = 0;

  virtual void updateGeometry(GlobalGeometryManager& geometry) const = 0;
  virtual void mapDirectionToGeometry(const DofBuffer& dq,
                                      GeometryBuffer& dx) const = 0;
  virtual void setInternalContacts(ContactTable contacts) = 0;
  virtual void scatterContactGradient(
      std::span<const GeometryPointId> points,
      const GeometryBuffer& pointGradient,
      DofBuffer& g) const = 0;
  virtual void applyContactHessian(const DofBuffer& dq,
                                   const ContactTable& contacts,
                                   DofBuffer& y) const = 0;
  virtual void visit(CpuSubsystemBackend& backend) = 0;
  virtual void visit(GpuSubsystemBackend& backend) = 0;
};

}  // namespace ksk::runtime
