#pragma once

#include <HairSim/rod.h>

#include <glm/glm.hpp>

#include <string>

namespace sim::hairsim {

class ImplicitRodIntegrator {
 public:
  explicit ImplicitRodIntegrator(Rod& rod) : rod_(rod) {}

  bool step(double dt, const glm::dvec3& gravity);

  [[nodiscard]] const RodEvaluation& lastEvaluation() const noexcept {
    return last_evaluation_;
  }
  [[nodiscard]] const std::string& diagnostic() const noexcept {
    return diagnostic_;
  }

 private:
  Rod& rod_;
  RodEvaluation last_evaluation_;
  std::string diagnostic_;
};

}  // namespace sim::hairsim
