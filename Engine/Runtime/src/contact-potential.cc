#include <Runtime/contact-potential.h>

#include <glm/geometric.hpp>

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ksk::runtime {
namespace {

struct PointPairEvaluation {
  GeometryPointId p0 = -1;
  GeometryPointId p1 = -1;
  glm::dvec3 normal{1.0, 0.0, 0.0};
  double distance = 0.0;
  double threshold = 0.0;
  double stiffness = 0.0;
  bool active = false;
};

PointPairEvaluation evaluatePointPair(const GlobalGeometryManager& geometry,
                                      const ContactStencil& stencil)
{
  const GeometryPointId p0 = stencil.geometryIds[0];
  const GeometryPointId p1 = stencil.geometryIds[1];
  if (!geometry.contains(p0) || !geometry.contains(p1)) {
    throw std::out_of_range("point-point contact stencil references invalid point");
  }

  const double threshold = stencil.dHat + stencil.thickness;
  if (threshold <= 0.0 || stencil.stiffness <= 0.0) {
    return {};
  }

  const glm::dvec3 diff =
      geometry.worldPosition(p0) - geometry.worldPosition(p1);
  const double distance = glm::length(diff);
  if (distance >= threshold) {
    return {};
  }

  const glm::dvec3 normal =
      distance > 1.0e-12 ? diff / distance : glm::dvec3(1.0, 0.0, 0.0);
  return PointPairEvaluation{
      .p0 = p0,
      .p1 = p1,
      .normal = normal,
      .distance = distance,
      .threshold = threshold,
      .stiffness = stencil.stiffness,
      .active = true,
  };
}

int appendGradientPoint(std::vector<GeometryPointId>& points,
                        std::vector<glm::dvec3>& gradients,
                        std::unordered_map<GeometryPointId, int>& offsets,
                        GeometryPointId point)
{
  const auto found = offsets.find(point);
  if (found != offsets.end()) {
    return found->second;
  }

  const int offset = static_cast<int>(points.size());
  offsets.emplace(point, offset);
  points.push_back(point);
  gradients.push_back(glm::dvec3(0.0));
  return offset;
}

}  // namespace

double evaluateContactEnergy(const GlobalGeometryManager& geometry,
                             const ContactTable& contacts)
{
  double energy = 0.0;
  for (const ContactStencil& stencil : contacts.stencils) {
    if (stencil.type != ContactCase::PP) {
      continue;
    }
    const PointPairEvaluation pair = evaluatePointPair(geometry, stencil);
    if (!pair.active) {
      continue;
    }

    const double gap = pair.threshold - pair.distance;
    energy += 0.5 * pair.stiffness * gap * gap;
  }
  return energy;
}

ContactPotentialGradient evaluateContactGradient(
    const GlobalGeometryManager& geometry,
    const ContactTable& contacts)
{
  std::vector<GeometryPointId> points;
  std::vector<glm::dvec3> gradients;
  std::unordered_map<GeometryPointId, int> offsets;

  for (const ContactStencil& stencil : contacts.stencils) {
    if (stencil.type != ContactCase::PP) {
      continue;
    }
    const PointPairEvaluation pair = evaluatePointPair(geometry, stencil);
    if (!pair.active) {
      continue;
    }

    const double gap = pair.threshold - pair.distance;
    const glm::dvec3 g0 = -pair.stiffness * gap * pair.normal;
    const int i0 = appendGradientPoint(points, gradients, offsets, pair.p0);
    const int i1 = appendGradientPoint(points, gradients, offsets, pair.p1);
    gradients[static_cast<size_t>(i0)] += g0;
    gradients[static_cast<size_t>(i1)] -= g0;
  }

  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(gradients)),
  };
}

}  // namespace ksk::runtime
