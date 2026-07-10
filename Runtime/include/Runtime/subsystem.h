#pragma once

#include <span>

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include <Runtime/contact-table.h>
#include <Runtime/dof-layout.h>
#include <Runtime/geometry-table.h>

namespace ksk::runtime {

class Subsystem {
 public:
  virtual ~Subsystem() = default;

  [[nodiscard]] virtual SubsystemId id() const noexcept = 0;
  [[nodiscard]] virtual DofRange dofRange() const noexcept = 0;

  virtual void appendGeometry(GeometryTable& geometry) = 0;
  virtual void writeState(Eigen::VectorXd& q, Eigen::VectorXd& qdot) const = 0;
  virtual void readState(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot) = 0;

  virtual void updateInternalConstraints(double time, double dt) = 0;
  virtual void prepareLocalOperator(double dt) = 0;
  virtual void assembleLocalGradient(Eigen::VectorXd& g) const = 0;
  virtual void applyLocalMatrix(const Eigen::VectorXd& x, Eigen::VectorXd& y) const = 0;
  virtual void solveLocalSystem(const Eigen::VectorXd& b, Eigen::VectorXd& x) const = 0;

  virtual void evaluateGeometryPositions(GeometryTable& geometry) const = 0;
  virtual void mapDirectionToGeometry(const Eigen::VectorXd& dq,
                                      std::span<glm::dvec3> dx) const = 0;
  virtual void scatterContactGradient(std::span<const GeometryPointId> points,
                                      std::span<const glm::dvec3> pointGradient,
                                      Eigen::VectorXd& g) const = 0;
  virtual void applyContactHessian(const Eigen::VectorXd& dq,
                                   const ContactTable& contacts,
                                   Eigen::VectorXd& y) const = 0;
};

}  // namespace ksk::runtime
