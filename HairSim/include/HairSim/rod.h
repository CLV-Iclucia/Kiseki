#pragma once

#include <Eigen/SparseCore>
#include <glm/glm.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ksk::hairsim {

using RodBlock = glm::dvec4;

struct RodState {
  std::vector<RodBlock> blocks;

  [[nodiscard]] size_t size() const noexcept { return blocks.size(); }
  [[nodiscard]] glm::dvec3 position(size_t i) const;
  void setPosition(size_t i, const glm::dvec3& value);
  [[nodiscard]] double theta(size_t i) const;
  void setTheta(size_t i, double value);
};

using RodVelocity = RodState;

struct RodRestState {
  std::vector<RodBlock> blocks;
  // For circular isotropic rods this stores rest curvature-binormal xyz.
  std::vector<glm::dvec4> curvature;
  // x/y: previous/next rest edge length, z: vertex dual length,
  // w: rest material twist theta_i - theta_{i-1} in the local Bishop gauge.
  std::vector<glm::dvec4> metrics;
};

struct RodMaterial {
  double density = 1300.0;
  double radius = 4e-5;
  double youngsModulus = 4e9;
  double shearModulus = 1.5e9;
  double rootStiffness = 2e6;
  bool pinRootTwist = false;

  [[nodiscard]] double area() const noexcept;
  [[nodiscard]] double areaMoment() const noexcept;
  [[nodiscard]] double polarMoment() const noexcept;
  [[nodiscard]] double axialStiffness() const noexcept;
  [[nodiscard]] double bendingStiffness() const noexcept;
  [[nodiscard]] double twistStiffness() const noexcept;
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
  std::vector<RodBlock> gradient;
  Eigen::SparseMatrix<double> hessian;
  bool valid = false;
  std::string diagnostic;
};

class Rod {
 public:
  Rod(std::vector<RodBlock> restBlocks, RodMaterial material = {});

  [[nodiscard]] const RodState& state() const noexcept { return state_; }
  [[nodiscard]] RodState& state() noexcept { return state_; }
  [[nodiscard]] const RodVelocity& velocity() const noexcept { return velocity_; }
  [[nodiscard]] RodVelocity& velocity() noexcept { return velocity_; }
  [[nodiscard]] const RodRestState& restState() const noexcept { return rest_; }
  [[nodiscard]] const RodMaterial& material() const noexcept { return material_; }

  void resetReferenceFrames();
  void transportReferenceFrames(const RodState& previous_state);
  [[nodiscard]] RodEvaluation evaluate(const glm::dvec3& gravity) const;
  [[nodiscard]] Eigen::VectorXd massDiagonal() const;

 private:
  RodState state_;
  RodVelocity velocity_;
  RodRestState rest_;
  std::vector<glm::dvec3> reference_directors_;
  RodMaterial material_;
};

}  // namespace ksk::hairsim
