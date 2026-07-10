#pragma once

#include <vector>

#include <glm/glm.hpp>

#include <Runtime/types.h>

namespace ksk::runtime {

struct GeometryPoint {
  GeometryPointId id;
  SubsystemId subsystem;
  int localSampleId = -1;
  glm::dvec3 x{0.0};
};

struct GeometryEdge {
  int id = -1;
  GeometryPointId p0;
  GeometryPointId p1;
  Real radius = 0.0;
};

struct GeometryTriangle {
  int id = -1;
  GeometryPointId p0;
  GeometryPointId p1;
  GeometryPointId p2;
};

struct GeometryTable {
  std::vector<GeometryPoint> points;
  std::vector<GeometryEdge> edges;
  std::vector<GeometryTriangle> triangles;

  [[nodiscard]] GeometryPointId appendPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position);
  [[nodiscard]] int appendEdge(GeometryPointId p0,
                               GeometryPointId p1,
                               Real radius = 0.0);
  [[nodiscard]] int appendTriangle(GeometryPointId p0,
                                   GeometryPointId p1,
                                   GeometryPointId p2);
  [[nodiscard]] bool contains(GeometryPointId point) const noexcept;
};

}  // namespace ksk::runtime
