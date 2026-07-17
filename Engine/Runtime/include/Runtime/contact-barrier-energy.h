#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <vector>

namespace ksk::runtime {

struct ContactPotentialGradient {
  std::vector<PointIdx> points;
  GeometryBuffer gradient;
};

[[nodiscard]] double computeContactEnergy(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactGradient(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactHessianProduct(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts,
    ConstGeometryView geometryDirection);

}  // namespace ksk::runtime
