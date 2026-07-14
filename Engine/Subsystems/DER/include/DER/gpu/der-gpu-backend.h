#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <span>

namespace ksk::der {

class DERSubsystem;

class DERGpuBackend {
 public:
  explicit DERGpuBackend(DERSubsystem& subsystem);

  void writeState(runtime::DofBuffer& q, runtime::DofBuffer& qdot) const;
  void readState(runtime::DofBuffer& q, runtime::DofBuffer& qdot);
  void beginStep(const runtime::DofBuffer& q,
                 const runtime::DofBuffer& qdot,
                 double dt);
  void acceptStep(const runtime::DofBuffer& q,
                  runtime::DofBuffer& qdot,
                  double dt);
  [[nodiscard]] double evaluateObjective(const runtime::DofBuffer& q,
                                         const runtime::DofBuffer& qdot,
                                         double dt);
  void updateInternalConstraints(double time, double dt);
  void prepareLocalOperator(double dt);
  void assembleLocalGradient(runtime::DofBuffer& g) const;
  void applyLocalMatrix(const runtime::DofBuffer& x,
                        runtime::DofBuffer& y) const;
  void solveLocalSystem(const runtime::DofBuffer& b,
                        runtime::DofBuffer& x) const;
  void mapDirectionToGeometry(const runtime::DofBuffer& dq,
                              runtime::GeometryBuffer& dx) const;
  void scatterContactGradient(
      std::span<const runtime::GeometryPointId> points,
      const runtime::GeometryBuffer& pointGradient,
      runtime::DofBuffer& g) const;
  void applyContactHessian(const runtime::DofBuffer& dq,
                           const runtime::ContactTable& contacts,
                           runtime::DofBuffer& y) const;

 private:
  [[noreturn]] static void unsupported(const char* operation);
  static void requireGPU(const runtime::DofBuffer& buffer,
                         const char* operation);
  static void requireGPU(const runtime::GeometryBuffer& buffer,
                         const char* operation);
  static void requireSameDevice(const runtime::DofBuffer& lhs,
                                const runtime::DofBuffer& rhs,
                                const char* operation);
  static void requireSameDevice(const runtime::DofBuffer& lhs,
                                const runtime::GeometryBuffer& rhs,
                                const char* operation);

  DERSubsystem& subsystem_;
};

}  // namespace ksk::der
