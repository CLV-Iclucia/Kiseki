#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <span>

namespace ksk::engine::fem {

class FEMSubsystem;

class FEMCPUBackend {
 public:
  explicit FEMCPUBackend(FEMSubsystem& subsystem);

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
  void applyContactGeometryHessianProduct(
      std::span<const runtime::PointIdx> gradientPoints,
      runtime::ConstGeometryView pointGradient,
      std::span<const runtime::PointIdx> productPoints,
      runtime::ConstGeometryView pointHessianProduct,
      runtime::ConstDofView localDq,
      runtime::DofView localY) const;
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const;

 private:
  [[nodiscard]] Eigen::VectorXd buildMassDiagonal() const;

  FEMSubsystem& subsystem_;
  Eigen::VectorXd step_start_;
  Eigen::VectorXd inertial_target_;
  Eigen::VectorXd mass_diagonal_;
  Eigen::SparseMatrix<double> local_matrix_;
  double step_dt_ = 0.0;
};

}  // namespace ksk::engine::fem
