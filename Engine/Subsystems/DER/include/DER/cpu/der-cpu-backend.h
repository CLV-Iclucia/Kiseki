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

  [[nodiscard]] const RodEvaluation& cachedEvaluation(int rod) const;

 private:
  [[nodiscard]] int vectorOffset(const Eigen::VectorXd& values,
                                 int localOffset) const;
  [[nodiscard]] Eigen::VectorXd gatherLocalVector(
      const Eigen::VectorXd& values) const;
  [[nodiscard]] Eigen::VectorXd gatherCurrentState() const;
  [[nodiscard]] RodEvaluation evaluateRod(const Rod& rod, int rodIndex);
  void addToVector(Eigen::VectorXd& values,
                   int localOffset,
                   double value) const;

  DERSubsystem& subsystem_;
  std::vector<RodEvaluation> evaluations_;
  Eigen::VectorXd step_start_;
  std::vector<std::vector<RodBlock>> step_previous_states_;
  Eigen::VectorXd inertial_target_;
  Eigen::VectorXd mass_diagonal_;
  double step_dt_ = 0.0;
  Eigen::SparseMatrix<double> local_matrix_;
};

}  // namespace ksk::der
