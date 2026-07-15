#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <vector>

namespace ksk::runtime {

struct ContactPotentialGradient {
  std::vector<GeometryPointId> points;
  GeometryBuffer gradient;
};

[[nodiscard]] double evaluateContactEnergy(
    const GlobalGeometryManager& geometry,
    const ContactTable& contacts);

[[nodiscard]] ContactPotentialGradient evaluateContactGradient(
    const GlobalGeometryManager& geometry,
    const ContactTable& contacts);

}  // namespace ksk::runtime
