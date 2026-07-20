#pragma once

#include <Runtime/global-geometry-manager.h>

#include <array>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace ksk::runtime {

struct GeometryTransferMap {
  ObjectId sourceObject = -1;
  GeometryRange points;
  GeometryRange edges;
  GeometryRange triangles;
  GeometryRange tets;
  std::vector<PointIdx> localToGlobalPoint;
  std::vector<int> localToGlobalEdge;
  std::vector<int> localToGlobalTriangle;
  std::vector<int> localToGlobalTet;
};

struct GeometryTransferInput {
  ObjectId sourceObject = -1;
  SubsystemId subsystem = -1;
  int localInstanceId = -1;
  int localSampleOffset = 0;
  std::span<const glm::dvec3> vertices;
  std::span<const glm::dvec3> positions;
  std::span<const std::array<int, 2>> edges;
  std::span<const std::array<int, 3>> triangles;
  std::span<const std::array<int, 4>> tets;
  Real radius = 0.0;
};

[[nodiscard]] GeometryTransferMap transferGeometry(
    GlobalGeometryManager& geometry,
    const GeometryTransferInput& input);

}  // namespace ksk::runtime
