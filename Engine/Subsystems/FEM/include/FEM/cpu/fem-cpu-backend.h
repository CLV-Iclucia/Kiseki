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
  [[nodiscard]] Eigen::VectorXd buildMassDiagonal() const;

  FEMSubsystem& subsystem_;
  Eigen::VectorXd step_start_;
  Eigen::VectorXd inertial_target_;
  Eigen::VectorXd mass_diagonal_;
  Eigen::SparseMatrix<double> local_matrix_;
  double step_dt_ = 0.0;
};

}  // namespace ksk::engine::fem
