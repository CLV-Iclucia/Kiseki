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
  Real thickness = 0.0;
};

struct ContactBatch {
  ContactCase type = ContactCase::PP;
  int first = 0;
  int count = 0;
};

struct ContactTable {
  std::vector<ContactStencil> stencils;
  std::vector<ContactBatch> batches;
};

}  // namespace ksk::runtime
