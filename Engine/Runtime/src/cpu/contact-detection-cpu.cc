#include <Runtime/contact-detection.h>

#include "../contact-detection-backends.h"

#include <Spatify/lbvh.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>

namespace ksk::runtime {
namespace {

constexpr int kMaxACCDIterations = 2000;
constexpr Real kACCDGapFraction = 0.1;

enum class PrimitiveKind {
  Point,
  Edge,
  Triangle,
};

enum class PointTriangleDistanceType {
  P_A,
  P_B,
  P_C,
  P_AB,
  P_BC,
  P_CA,
  P_ABC,
};

enum class EdgeEdgeDistanceType {
  A_C,
  A_D,
  B_C,
  B_D,
  AB_C,
  AB_D,
  A_CD,
  B_CD,
  AB_CD,
};

struct PrimitiveRef {
  PrimitiveKind kind = PrimitiveKind::Point;
  int id = -1;
  Real radius = 0.0;
  spatify::BBox<Real, 3> bounds;
};

struct PrimitiveAccessor {
  using CoordType = Real;

  const std::vector<PrimitiveRef>* primitives = nullptr;

  [[nodiscard]] spatify::BBox<Real, 3> bbox(int index) const
  {
    return primitives->at(static_cast<size_t>(index)).bounds;
  }

  [[nodiscard]] int size() const
  {
    return static_cast<int>(primitives->size());
  }
};

spatify::BBox<Real, 3> toSpatifyBounds(const GeometryBounds& bounds)
{
  return spatify::BBox<Real, 3>(bounds.lower, bounds.upper);
}

spatify::BBox<Real, 3> toSpatifyBounds(const GeometryBounds& bounds,
                                       Real radius)
{
  spatify::BBox<Real, 3> result = toSpatifyBounds(bounds);
  if (radius > 0.0) {
    result.dilate(radius);
  }
  return result;
}

ContactStencil makeStencil(ContactCase type,
                           std::array<int, 4> ids,
                           const ContactDetectionConfig& config,
                           Real primitive_thickness = 0.0)
{
  return ContactStencil{
      .type = type,
      .geometryIds = ids,
      .dHat = config.dHat,
      .stiffness = config.stiffness,
      .thickness = config.thickness + primitive_thickness,
  };
}

Real distanceSqrPointPoint(const glm::dvec3& a, const glm::dvec3& b)
{
  const glm::dvec3 d = a - b;
  return glm::dot(d, d);
}

Real distanceSqrPointLine(const glm::dvec3& p,
                          const glm::dvec3& a,
                          const glm::dvec3& b)
{
  const glm::dvec3 e = b - a;
  const Real e_len2 = glm::dot(e, e);
  if (e_len2 == 0.0) {
    return distanceSqrPointPoint(p, a);
  }
  const glm::dvec3 c = glm::cross(e, p - a);
  return glm::dot(c, c) / e_len2;
}

Real distanceSqrPointPlane(const glm::dvec3& p,
                           const glm::dvec3& a,
                           const glm::dvec3& b,
                           const glm::dvec3& c)
{
  const glm::dvec3 normal = glm::cross(b - a, c - a);
  const Real n_len2 = glm::dot(normal, normal);
  if (n_len2 == 0.0) {
    return std::min({distanceSqrPointLine(p, a, b),
                     distanceSqrPointLine(p, b, c),
                     distanceSqrPointLine(p, c, a)});
  }
  const Real d = glm::dot(p - a, normal);
  return d * d / n_len2;
}

Real distanceSqrLineLine(const glm::dvec3& a0,
                         const glm::dvec3& a1,
                         const glm::dvec3& b0,
                         const glm::dvec3& b1)
{
  const glm::dvec3 normal = glm::cross(a1 - a0, b1 - b0);
  const Real n_len2 = glm::dot(normal, normal);
  if (n_len2 == 0.0) {
    return distanceSqrPointLine(a0, b0, b1);
  }
  const Real d = glm::dot(a0 - b0, normal);
  return d * d / n_len2;
}

void pointTriangleEdgeTest(const glm::dvec3& from,
                           const glm::dvec3& to,
                           const glm::dvec3& point,
                           const glm::dvec3& normal,
                           Real& s,
                           Real& t)
{
  const glm::dvec3 e = to - from;
  const glm::dvec3 n = glm::cross(e, normal);
  const Real a00 = glm::dot(e, e);
  const Real a01 = glm::dot(e, n);
  const Real a11 = glm::dot(n, n);
  const Real b0 = glm::dot(e, point - from);
  const Real b1 = glm::dot(n, point - from);
  const Real det = a00 * a11 - a01 * a01;
  if (det == 0.0) {
    s = 0.0;
    t = -1.0;
    return;
  }
  s = (a11 * b0 - a01 * b1) / det;
  t = (a00 * b1 - a01 * b0) / det;
}

PointTriangleDistanceType decidePointTriangleDistanceType(
    const glm::dvec3& p,
    const glm::dvec3& a,
    const glm::dvec3& b,
    const glm::dvec3& c)
{
  const glm::dvec3 normal = glm::cross(b - a, c - a);
  if (glm::dot(normal, normal) == 0.0) {
    const Real da = distanceSqrPointLine(p, a, b);
    const Real db = distanceSqrPointLine(p, b, c);
    const Real dc = distanceSqrPointLine(p, c, a);
    if (da <= db && da <= dc) {
      return PointTriangleDistanceType::P_AB;
    }
    if (db <= dc) {
      return PointTriangleDistanceType::P_BC;
    }
    return PointTriangleDistanceType::P_CA;
  }

  Real s0 = 0.0;
  Real t0 = 0.0;
  pointTriangleEdgeTest(a, b, p, normal, s0, t0);
  if (s0 > 0.0 && s0 < 1.0 && t0 >= 0.0) {
    return PointTriangleDistanceType::P_AB;
  }

  Real s1 = 0.0;
  Real t1 = 0.0;
  pointTriangleEdgeTest(b, c, p, normal, s1, t1);
  if (s1 > 0.0 && s1 < 1.0 && t1 >= 0.0) {
    return PointTriangleDistanceType::P_BC;
  }

  Real s2 = 0.0;
  Real t2 = 0.0;
  pointTriangleEdgeTest(c, a, p, normal, s2, t2);
  if (s2 > 0.0 && s2 < 1.0 && t2 >= 0.0) {
    return PointTriangleDistanceType::P_CA;
  }

  if (s0 <= 0.0 && s2 >= 1.0) {
    return PointTriangleDistanceType::P_A;
  }
  if (s1 <= 0.0 && s0 >= 1.0) {
    return PointTriangleDistanceType::P_B;
  }
  if (s2 <= 0.0 && s1 >= 1.0) {
    return PointTriangleDistanceType::P_C;
  }
  return PointTriangleDistanceType::P_ABC;
}

EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(
    const glm::dvec3& ea0,
    const glm::dvec3& ea1,
    const glm::dvec3& eb0,
    const glm::dvec3& eb1)
{
  const glm::dvec3 ea = ea1 - ea0;
  const Real ea_sqr = glm::dot(ea, ea);
  if (ea_sqr == 0.0) {
    return EdgeEdgeDistanceType::A_C;
  }
  const Real alpha = glm::dot(eb0 - ea0, ea) / ea_sqr;
  const Real beta = glm::dot(eb1 - ea0, ea) / ea_sqr;

  int eac = 0;
  int ebc = 0;
  if (alpha < 0.0) {
    eac = (0.0 <= beta && beta <= 1.0) ? 2 : 0;
    ebc = (beta <= alpha) ? 0 : (beta <= 1.0 ? 1 : 2);
  } else if (alpha > 1.0) {
    eac = (0.0 <= beta && beta <= 1.0) ? 2 : 1;
    ebc = (beta >= alpha) ? 0 : (0.0 <= beta ? 1 : 2);
  } else {
    eac = 2;
    ebc = 0;
  }

  if (ebc < 2) {
    const int code = (eac << 1) | ebc;
    switch (code) {
      case 0:
        return EdgeEdgeDistanceType::A_C;
      case 1:
        return EdgeEdgeDistanceType::A_D;
      case 2:
        return EdgeEdgeDistanceType::B_C;
      case 3:
        return EdgeEdgeDistanceType::B_D;
      case 4:
        return EdgeEdgeDistanceType::AB_C;
      default:
        return EdgeEdgeDistanceType::AB_D;
    }
  }

  return eac == 0 ? EdgeEdgeDistanceType::A_CD
                  : EdgeEdgeDistanceType::B_CD;
}

EdgeEdgeDistanceType decideEdgeEdgeDistanceType(const glm::dvec3& ea0,
                                                const glm::dvec3& ea1,
                                                const glm::dvec3& eb0,
                                                const glm::dvec3& eb1)
{
  constexpr Real parallel_threshold = 1.0e-20;
  const glm::dvec3 u = ea1 - ea0;
  const glm::dvec3 v = eb1 - eb0;
  const glm::dvec3 w = ea0 - eb0;

  const Real a = glm::dot(u, u);
  const Real b = glm::dot(u, v);
  const Real c = glm::dot(v, v);
  const Real d = glm::dot(u, w);
  const Real e = glm::dot(v, w);
  const Real det = a * c - b * b;

  if (a == 0.0 && c == 0.0) {
    return EdgeEdgeDistanceType::A_C;
  }
  if (a == 0.0) {
    return EdgeEdgeDistanceType::A_CD;
  }
  if (c == 0.0) {
    return EdgeEdgeDistanceType::AB_C;
  }

  const Real parallel_tolerance = parallel_threshold * std::max(1.0, a * c);
  const glm::dvec3 uxv = glm::cross(u, v);
  if (glm::dot(uxv, uxv) < parallel_tolerance) {
    return decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
  }

  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
  const Real sN = b * e - c * d;
  Real tN = 0.0;
  Real tD = 0.0;
  if (sN <= 0.0) {
    tN = e;
    tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= det) {
    tN = e + b;
    tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = a * e - b * d;
    tD = det;
  }

  if (tN <= 0.0) {
    if (-d <= 0.0) {
      return EdgeEdgeDistanceType::A_C;
    }
    return -d >= a ? EdgeEdgeDistanceType::B_C
                   : EdgeEdgeDistanceType::AB_C;
  }
  if (tN >= tD) {
    if ((-d + b) <= 0.0) {
      return EdgeEdgeDistanceType::A_D;
    }
    return (-d + b) >= a ? EdgeEdgeDistanceType::B_D
                         : EdgeEdgeDistanceType::AB_D;
  }
  return default_case;
}

Real pointTriangleDistanceSqrByType(PointTriangleDistanceType type,
                                    const glm::dvec3& p,
                                    const glm::dvec3& a,
                                    const glm::dvec3& b,
                                    const glm::dvec3& c)
{
  switch (type) {
    case PointTriangleDistanceType::P_A:
      return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:
      return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:
      return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:
      return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:
      return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:
      return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:
      return distanceSqrPointPlane(p, a, b, c);
  }
  return 0.0;
}

Real edgeEdgeDistanceSqrByType(EdgeEdgeDistanceType type,
                               const glm::dvec3& a0,
                               const glm::dvec3& a1,
                               const glm::dvec3& b0,
                               const glm::dvec3& b1)
{
  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      return distanceSqrPointPoint(a0, b0);
    case EdgeEdgeDistanceType::A_D:
      return distanceSqrPointPoint(a0, b1);
    case EdgeEdgeDistanceType::B_C:
      return distanceSqrPointPoint(a1, b0);
    case EdgeEdgeDistanceType::B_D:
      return distanceSqrPointPoint(a1, b1);
    case EdgeEdgeDistanceType::AB_C:
      return distanceSqrPointLine(b0, a0, a1);
    case EdgeEdgeDistanceType::AB_D:
      return distanceSqrPointLine(b1, a0, a1);
    case EdgeEdgeDistanceType::A_CD:
      return distanceSqrPointLine(a0, b0, b1);
    case EdgeEdgeDistanceType::B_CD:
      return distanceSqrPointLine(a1, b0, b1);
    case EdgeEdgeDistanceType::AB_CD:
      return distanceSqrLineLine(a0, a1, b0, b1);
  }
  return 0.0;
}

std::optional<Real> runACCD(bool edgeEdge,
                            std::array<glm::dvec3, 4> x,
                            std::array<glm::dvec3, 4> u,
                            Real toi,
                            Real reserved_distance)
{
  const glm::dvec3 mean_step = (u[0] + u[1] + u[2] + u[3]) * 0.25;
  for (glm::dvec3& step : u) {
    step -= mean_step;
  }

  const Real lp =
      edgeEdge ? std::max(glm::length(u[0]), glm::length(u[1])) +
                     std::max(glm::length(u[2]), glm::length(u[3]))
               : glm::length(u[0]) +
                     std::max({glm::length(u[1]),
                               glm::length(u[2]),
                               glm::length(u[3])});
  if (lp == 0.0) {
    return std::nullopt;
  }

  auto distance = [&]() -> Real {
    if (edgeEdge) {
      const EdgeEdgeDistanceType type =
          decideEdgeEdgeDistanceType(x[0], x[1], x[2], x[3]);
      return std::sqrt(
          edgeEdgeDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
    }
    const PointTriangleDistanceType type =
        decidePointTriangleDistanceType(x[0], x[1], x[2], x[3]);
    return std::sqrt(
        pointTriangleDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
  };

  Real dis = distance();
  if (dis <= reserved_distance) {
    return 0.0;
  }
  const Real initial_gap = dis - reserved_distance;
  const Real gap = kACCDGapFraction * initial_gap;
  Real t = 0.0;
  Real dt = (1.0 - kACCDGapFraction) * (initial_gap / lp);

  for (int iteration = 0; iteration < kMaxACCDIterations; ++iteration) {
    for (int i = 0; i < 4; ++i) {
      x[static_cast<size_t>(i)] += u[static_cast<size_t>(i)] * dt;
    }
    dis = distance();
    if (dis - reserved_distance < gap + 1.0e-10) {
      return t == 0.0 ? dt : t;
    }
    t += dt;
    if (t > toi) {
      return std::nullopt;
    }
    dt = 0.9 * (dis - reserved_distance) / lp;
  }
  return t > 0.0 ? std::optional<Real>(t) : std::nullopt;
}

bool shouldKeepByDistanceOrCCD(Real distance_sqr,
                               std::optional<Real> ccd_toi,
                               const ContactDetectionConfig& config,
                               Real primitive_thickness)
{
  const Real threshold =
      std::max(config.detectionDistance, config.dHat) + config.thickness +
      primitive_thickness;
  return distance_sqr <= threshold * threshold ||
         (ccd_toi && *ccd_toi <= config.toi);
}

std::optional<ContactStencil> makePointTriangleStencil(
    GeometryPointId point,
    const std::array<GeometryPointId, 3>& triangle,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config,
    Real primitive_thickness)
{
  const glm::dvec3 p = geometry.worldPosition(point);
  const glm::dvec3 a = geometry.worldPosition(triangle[0]);
  const glm::dvec3 b = geometry.worldPosition(triangle[1]);
  const glm::dvec3 c = geometry.worldPosition(triangle[2]);
  const PointTriangleDistanceType type =
      decidePointTriangleDistanceType(p, a, b, c);
  const Real distance_sqr = pointTriangleDistanceSqrByType(type, p, a, b, c);
  const std::optional<Real> toi =
      runACCD(false,
              {p, a, b, c},
              {direction[point],
               direction[triangle[0]],
               direction[triangle[1]],
               direction[triangle[2]]},
              config.toi,
              config.thickness + primitive_thickness);
  if (!shouldKeepByDistanceOrCCD(
          distance_sqr, toi, config, primitive_thickness)) {
    return std::nullopt;
  }

  switch (type) {
    case PointTriangleDistanceType::P_A:
      return makeStencil(
          ContactCase::PP, {point, triangle[0], -1, -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_B:
      return makeStencil(
          ContactCase::PP, {point, triangle[1], -1, -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_C:
      return makeStencil(
          ContactCase::PP, {point, triangle[2], -1, -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_AB:
      return makeStencil(
          ContactCase::PE, {point, triangle[0], triangle[1], -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_BC:
      return makeStencil(
          ContactCase::PE, {point, triangle[1], triangle[2], -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_CA:
      return makeStencil(
          ContactCase::PE, {point, triangle[2], triangle[0], -1}, config, primitive_thickness);
    case PointTriangleDistanceType::P_ABC:
      return makeStencil(
          ContactCase::PT,
          {point, triangle[0], triangle[1], triangle[2]},
          config,
          primitive_thickness);
  }
  return std::nullopt;
}

std::optional<ContactStencil> makeEdgeEdgeStencil(
    const std::array<GeometryPointId, 2>& first,
    const std::array<GeometryPointId, 2>& second,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config,
    Real primitive_thickness)
{
  const glm::dvec3 a0 = geometry.worldPosition(first[0]);
  const glm::dvec3 a1 = geometry.worldPosition(first[1]);
  const glm::dvec3 b0 = geometry.worldPosition(second[0]);
  const glm::dvec3 b1 = geometry.worldPosition(second[1]);
  const EdgeEdgeDistanceType type = decideEdgeEdgeDistanceType(a0, a1, b0, b1);
  const Real distance_sqr =
      edgeEdgeDistanceSqrByType(type, a0, a1, b0, b1);
  const std::optional<Real> toi =
      runACCD(true,
              {a0, a1, b0, b1},
              {direction[first[0]],
               direction[first[1]],
               direction[second[0]],
               direction[second[1]]},
              config.toi,
              config.thickness + primitive_thickness);
  if (!shouldKeepByDistanceOrCCD(
          distance_sqr, toi, config, primitive_thickness)) {
    return std::nullopt;
  }

  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      return makeStencil(
          ContactCase::PP, {first[0], second[0], -1, -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::A_D:
      return makeStencil(
          ContactCase::PP, {first[0], second[1], -1, -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::B_C:
      return makeStencil(
          ContactCase::PP, {first[1], second[0], -1, -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::B_D:
      return makeStencil(
          ContactCase::PP, {first[1], second[1], -1, -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::AB_C:
      return makeStencil(
          ContactCase::PE, {second[0], first[0], first[1], -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::AB_D:
      return makeStencil(
          ContactCase::PE, {second[1], first[0], first[1], -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::A_CD:
      return makeStencil(
          ContactCase::PE, {first[0], second[0], second[1], -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::B_CD:
      return makeStencil(
          ContactCase::PE, {first[1], second[0], second[1], -1}, config, primitive_thickness);
    case EdgeEdgeDistanceType::AB_CD:
      return makeStencil(
          ContactCase::EE,
          {first[0], first[1], second[0], second[1]},
          config,
          primitive_thickness);
  }
  return std::nullopt;
}

std::optional<ContactStencil> makeStencil(
    const PrimitiveRef& a,
    const PrimitiveRef& b,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config)
{
  if (a.kind == PrimitiveKind::Point && b.kind == PrimitiveKind::Point) {
    return makeStencil(
        ContactCase::PP, {a.id, b.id, -1, -1}, config, a.radius + b.radius);
  }

  if (a.kind == PrimitiveKind::Point && b.kind == PrimitiveKind::Edge) {
    const auto edge = geometry.globalEdge(b.id);
    if (edge[0] == a.id || edge[1] == a.id) {
      return std::nullopt;
    }
    return makeStencil(
        ContactCase::PE, {a.id, edge[0], edge[1], -1}, config, a.radius + b.radius);
  }

  if (a.kind == PrimitiveKind::Edge && b.kind == PrimitiveKind::Point) {
    return makeStencil(b, a, geometry, direction, config);
  }

  if (a.kind == PrimitiveKind::Point && b.kind == PrimitiveKind::Triangle) {
    if (geometry.triangleContainsPoint(b.id, GeometryPointId{a.id})) {
      return std::nullopt;
    }
    const auto triangle = geometry.globalTriangle(b.id);
    return makePointTriangleStencil(
        a.id, triangle, geometry, direction, config, a.radius + b.radius);
  }

  if (a.kind == PrimitiveKind::Triangle && b.kind == PrimitiveKind::Point) {
    return makeStencil(b, a, geometry, direction, config);
  }

  if (a.kind == PrimitiveKind::Edge && b.kind == PrimitiveKind::Edge) {
    if (geometry.edgesAdjacent(a.id, b.id)) {
      return std::nullopt;
    }
    const auto first = geometry.globalEdge(a.id);
    const auto second = geometry.globalEdge(b.id);
    return makeEdgeEdgeStencil(
        first, second, geometry, direction, config, a.radius + b.radius);
  }

  return std::nullopt;
}

void rebuildBatches(ContactTable& table)
{
  table.batches.clear();
  std::stable_sort(table.stencils.begin(),
                   table.stencils.end(),
                   [](const ContactStencil& a, const ContactStencil& b) {
                     return static_cast<int>(a.type) < static_cast<int>(b.type);
                   });

  int first = 0;
  while (first < static_cast<int>(table.stencils.size())) {
    const ContactCase type = table.stencils[static_cast<size_t>(first)].type;
    int end = first + 1;
    while (end < static_cast<int>(table.stencils.size()) &&
           table.stencils[static_cast<size_t>(end)].type == type) {
      ++end;
    }
    table.batches.push_back(ContactBatch{
        .type = type,
        .first = first,
        .count = end - first,
    });
    first = end;
  }
}

std::vector<PrimitiveRef> buildPrimitives(const GlobalGeometryManager& geometry,
                                          const GeometryBuffer& direction,
                                          Real toi)
{
  if (!direction.isCPU()) {
    throw std::runtime_error(
        "CPU contact detection requires a CPU geometry direction buffer");
  }
  if (direction.cpu().size() < static_cast<size_t>(geometry.pointCount())) {
    throw std::invalid_argument(
        "geometry direction buffer is smaller than geometry");
  }

  std::vector<PrimitiveRef> primitives;
  primitives.reserve(static_cast<size_t>(geometry.pointCount() +
                                         geometry.edgeCount() +
                                         geometry.triangleCount()));
  for (int point = 0; point < geometry.pointCount(); ++point) {
    primitives.push_back(PrimitiveRef{
        .kind = PrimitiveKind::Point,
        .id = point,
        .radius = 0.0,
        .bounds = toSpatifyBounds(geometry.trajectoryPointBounds(
            GeometryPointId{point}, direction.cpu(), toi)),
    });
  }
  for (int edge = 0; edge < geometry.edgeCount(); ++edge) {
    const Real radius = geometry.edges[static_cast<size_t>(edge)].radius;
    primitives.push_back(PrimitiveRef{
        .kind = PrimitiveKind::Edge,
        .id = edge,
        .radius = radius,
        .bounds = toSpatifyBounds(
            geometry.trajectoryEdgeBounds(edge, direction.cpu(), toi),
            radius),
    });
  }
  for (int triangle = 0; triangle < geometry.triangleCount(); ++triangle) {
    const Real radius =
        geometry.triangles[static_cast<size_t>(triangle)].radius;
    primitives.push_back(PrimitiveRef{
        .kind = PrimitiveKind::Triangle,
        .id = triangle,
        .radius = radius,
        .bounds = toSpatifyBounds(
            geometry.trajectoryTriangleBounds(triangle, direction.cpu(), toi),
            radius),
    });
  }
  return primitives;
}

ContactTable& internalTableFor(RoutedContactTables& routed,
                               SubsystemId subsystem)
{
  for (SubsystemContactTable& entry : routed.internalContacts) {
    if (entry.subsystem == subsystem) {
      return entry.contacts;
    }
  }
  routed.internalContacts.push_back(SubsystemContactTable{
      .subsystem = subsystem,
  });
  return routed.internalContacts.back().contacts;
}

int stencilPointCount(ContactCase type)
{
  switch (type) {
    case ContactCase::PP:
      return 2;
    case ContactCase::PE:
      return 3;
    case ContactCase::PT:
    case ContactCase::EE:
      return 4;
  }
  return 0;
}

void routeStencil(RoutedContactTables& routed,
                  const GlobalGeometryManager& geometry,
                  const ContactStencil& stencil)
{
  const int point_count = stencilPointCount(stencil.type);
  const GeometryStencilInfo info = geometry.classify(
      std::span<const GeometryPointId>(stencil.geometryIds.data(),
                                       static_cast<size_t>(point_count)));
  if (info.valid && !info.hasCollider && info.subsystemCount == 1) {
    internalTableFor(routed, info.subsystems[0]).stencils.push_back(stencil);
    return;
  }
  routed.globalContacts.stencils.push_back(stencil);
}

RoutedContactTables detectContactsAlongDirectionCPUImpl(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }
  RoutedContactTables routed;
  std::vector<PrimitiveRef> primitives =
      buildPrimitives(geometry, geometryDirection, config.toi);
  if (primitives.empty()) {
    return routed;
  }

  spatify::LBVH<Real> bvh;
  bvh.update(PrimitiveAccessor{.primitives = &primitives});

  const Real query_dilation =
      std::max(config.detectionDistance, config.dHat) + config.thickness;
  for (int query_index = 0; query_index < static_cast<int>(primitives.size());
       ++query_index) {
    const PrimitiveRef& query_primitive =
        primitives[static_cast<size_t>(query_index)];
    const spatify::BBox<Real, 3> query_bounds =
        query_primitive.bounds.dilate(query_dilation);

    bvh.runSpatialQuery(
        [&](int candidate_index) -> bool {
          if (candidate_index <= query_index) {
            return false;
          }
          const std::optional<ContactStencil> stencil = makeStencil(
              query_primitive,
              primitives[static_cast<size_t>(candidate_index)],
              geometry,
              geometryDirection.cpu(),
              config);
          if (stencil) {
            routeStencil(routed, geometry, *stencil);
          }
          return stencil.has_value();
        },
        [&](const spatify::BBox<Real, 3>& bounds) -> bool {
          return query_bounds.overlap(bounds);
        });
  }

  rebuildBatches(routed.globalContacts);
  for (SubsystemContactTable& entry : routed.internalContacts) {
    rebuildBatches(entry.contacts);
  }
  return routed;
}

}  // namespace

namespace detail {

ContactDetectionOutput detectContactsAlongDirectionCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  return detectContactsAlongDirectionCPUImpl(geometry, geometryDirection, config);
}

}  // namespace detail

}  // namespace ksk::runtime
