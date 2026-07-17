#pragma once

#include <DER/rod.h>
#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <span>
#include <vector>

namespace ksk::der {

class DERSubsystem;

class DERCPUBackend {
 public:
  explicit DERCPUBackend(DERSubsystem& subsystem);

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

  [[nodiscard]] const RodEvaluation& cachedEvaluation(int rod) const;

 private:
  [[nodiscard]] Eigen::VectorXd gatherLocalVector(
      runtime::ConstDofView values) const;
  [[nodiscard]] Eigen::VectorXd gatherCurrentState() const;
  [[nodiscard]] RodEvaluation evaluateRod(const Rod& rod, int rodIndex);
  void addToVector(runtime::DofView values,
                   int localOffset,
                   double value) const;

  DERSubsystem& subsystem_;
  std::vector<RodEvaluation> evaluations_;
  Eigen::VectorXd step_start_;
  std::vector<std::vector<RodDof>> step_previous_states_;
  Eigen::VectorXd inertial_target_;
  Eigen::VectorXd mass_diagonal_;
  double step_dt_ = 0.0;
  Eigen::SparseMatrix<double> local_matrix_;
};

}  // namespace ksk::der
