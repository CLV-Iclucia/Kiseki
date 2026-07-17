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

  void writeState(runtime::DofView q, runtime::DofView qdot) const;
  void readState(runtime::ConstDofView q, runtime::ConstDofView qdot);
  void beginStep(runtime::ConstDofView q,
                 runtime::ConstDofView qdot,
                 double dt);
  void acceptStep(runtime::ConstDofView q,
                  runtime::DofView qdot,
                  double dt);
  [[nodiscard]] double evaluateObjective(runtime::ConstDofView q,
                                         runtime::ConstDofView qdot,
                                         double dt);
  void updateInternalConstraints(double time, double dt);
  void prepareLocalOperator(double dt);
  void assembleLocalGradient(runtime::DofView g) const;
  void applyLocalMatrix(runtime::ConstDofView x,
                        runtime::DofView y) const;
  void solveLocalSystem(runtime::ConstDofView b,
                        runtime::DofView x) const;
  void mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                              runtime::GeometryView globalDx) const;
  void scatterContactGradient(
      std::span<const runtime::PointIdx> points,
      runtime::ConstGeometryView pointGradient,
      runtime::DofView g) const;
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const;

 private:
  [[noreturn]] static void unsupported(const char* operation);
  static void requireGPU(runtime::ConstDofView view,
                         const char* operation);
  static void requireGPU(runtime::DofView view,
                         const char* operation);
  static void requireGPU(runtime::ConstGeometryView view,
                         const char* operation);
  static void requireGPU(runtime::GeometryView view,
                         const char* operation);
  static void requireSameDevice(runtime::ConstDofView lhs,
                                runtime::ConstDofView rhs,
                                const char* operation);
  static void requireSameDevice(runtime::ConstDofView lhs,
                                runtime::ConstGeometryView rhs,
                                const char* operation);

  DERSubsystem& subsystem_;
};

}  // namespace ksk::der
