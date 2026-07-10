//
// Created by creeper on 5/23/24.
//

#pragma once

#include <fem/system.h>
#include <Core/json.h>

namespace ksk::fem {
struct Integrator {
  virtual void step(Real dt) = 0;
  explicit Integrator(System &system_) : system_to_integrate(system_) {}
  System &system_to_integrate;
  [[nodiscard]] System &system() const { return system_to_integrate; }
  virtual ~Integrator() = default;

  static std::unique_ptr<Integrator> create(System& system, const core::JsonNode& json);
};

std::unique_ptr<Integrator> createIntegrator(System& system, const core::JsonNode &node);

}
