#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Runtime/types.h>

namespace ksk::runtime {

enum class ContactCase : std::uint16_t {
  PP,
  PE,
  PT,
  EE,
};

struct ContactStencil {
  ContactCase type = ContactCase::PP;
  std::array<int, 4> geometryIds{-1, -1, -1, -1};
  Real dHat = 0.0;
  Real stiffness = 0.0;
  // Total surface offset reserved from geometric distance before evaluating
  // the barrier, computed from the involved primitive radii/thicknesses.
  Real thickness = 0.0;
};

  using ContactStencils = std::vector<ContactStencil>;

}  // namespace ksk::runtime
