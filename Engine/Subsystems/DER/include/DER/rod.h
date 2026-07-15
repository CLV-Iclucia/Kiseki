#pragma once

#include <Eigen/SparseCore>
#include <glm/glm.hpp>

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ksk::der {

using RodDof = glm::dvec4;

struct RodState {
  std::span<RodDof> blocks;

  [[nodiscard]] size_t size() const noexcept { return blocks.size(); }
  [[nodiscard]] glm::dvec3 position(size_t index) const;
  void setPosition(size_t index, const glm::dvec3& value);
  [[nodiscard]] double theta(size_t index) const;
  void setTheta(size_t index, double value);
};

using RodVelocity = RodState;

struct RodRestState {
  std::vector<RodDof> blocks;
  std::vector<glm::dvec4> curvature;
  std::vector<glm::dvec4> metrics;
};

struct RodMaterial {
  double density = 1300.0;
  double radius = 4e-5;
  double youngsModulus = 4e9;
  double shearModulus = 1.5e9;

  [[nodiscard]] double area() const noexcept;
  [[nodiscard]] double areaMoment() const noexcept;
  [[nodiscard]] double polarMoment() const noexcept;
  [[nodiscard]] double axialStiffness() const noexcept;
  [[nodiscard]] double bendingStiffness() const noexcept;
  [[nodiscard]] double twistStiffness() const noexcept;
};

enum class RodConstraintProperty
{
  X,
  Y,
  Z,
  Twist,
};

struct RodConstraint {
  RodConstraintProperty property = RodConstraintProperty::X;
  size_t sample = 0;
  double target = 0.0;
  double stiffness = 0.0;
};

struct RodEnergyComponents {
  double stretching = 0.0;
  double bending = 0.0;
  double twisting = 0.0;
  double gravity = 0.0;
  double constraint = 0.0;

  [[nodiscard]] double total() const noexcept;
};

struct RodEvaluation {
  RodEnergyComponents energy;
  std::vector<RodDof> gradient;
  Eigen::SparseMatrix<double> hessian;
  bool valid = false;
  std::string diagnostic;
};

class Rod {
 public:
  Rod(std::vector<RodDof> restBlocks, RodMaterial material = {});
  Rod(const Rod& other);
  Rod(Rod&& other) noexcept;
  Rod& operator=(const Rod& other);
  Rod& operator=(Rod&& other) noexcept;

  [[nodiscard]] const RodState& state() const noexcept { return state_; }
  [[nodiscard]] RodState& state() noexcept { return state_; }
  [[nodiscard]] const RodVelocity& velocity() const noexcept { return velocity_; }
  [[nodiscard]] RodVelocity& velocity() noexcept { return velocity_; }
  [[nodiscard]] const RodRestState& restState() const noexcept { return rest_; }
  [[nodiscard]] const RodMaterial& material() const noexcept { return material_; }

  [[nodiscard]] double referenceTwist(size_t vertex) const;
  [[nodiscard]] double materialTwist(size_t vertex) const;
  void setRootPositionPinned(bool pinned) noexcept;
  void setTipPositionPinned(bool pinned) noexcept;
  void setTerminalThetaTarget(std::optional<double> target) noexcept;
  void addConstraint(RodConstraint constraint);
  void clearConstraints() noexcept;
  [[nodiscard]] std::span<const RodConstraint> constraints() const noexcept;
  void bindState(std::span<RodDof> stateBlocks,
                 std::span<RodDof> velocityBlocks);

  void resetReferenceFrames();
  void transportReferenceFrames(const RodState& previousState);
  [[nodiscard]] RodEvaluation evaluate(const glm::dvec3& gravity) const;
  [[nodiscard]] RodEvaluation evaluate(const glm::dvec3& gravity,
                                       const RodState& previousState) const;
  [[nodiscard]] Eigen::VectorXd massDiagonal() const;

 private:
  std::vector<RodDof> state_storage_;
  std::vector<RodDof> velocity_storage_;
  RodState state_;
  RodVelocity velocity_;
  RodRestState rest_;
  std::vector<glm::dvec3> reference_directors_;
  RodMaterial material_;
  bool pin_root_position_ = false;
  bool pin_tip_position_ = false;
  std::optional<double> terminal_theta_target_;
  std::vector<RodConstraint> constraints_;

  [[nodiscard]] RodEvaluation evaluate(
      const glm::dvec3& gravity,
      const std::vector<double>& referenceTwists) const;
  void bindOwnedState() noexcept;
};

}  // namespace ksk::der
