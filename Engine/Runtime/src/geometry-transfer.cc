#include <Runtime/geometry-transfer.h>

#include <stdexcept>

namespace ksk::runtime {
namespace {

void validateIndex(int index, int vertexCount, const char* label)
{
  if (index < 0 || index >= vertexCount) {
    throw std::out_of_range(label);
  }
}

}  // namespace

GeometryTransferMap transferGeometry(GlobalGeometryManager& geometry,
                                     const GeometryTransferInput& input)
{
  if (input.subsystem < 0) {
    throw std::invalid_argument("geometry transfer subsystem is invalid");
  }
  if (input.vertices.empty()) {
    throw std::invalid_argument("geometry transfer has no vertices");
  }
  if (!input.positions.empty() && input.positions.size() != input.vertices.size()) {
    throw std::invalid_argument(
        "geometry transfer positions must match vertex count");
  }

  const int vertex_count = static_cast<int>(input.vertices.size());

#ifndef NDEBUG
  for (const auto& edge : input.edges) {
    validateIndex(edge[0], vertex_count, "geometry transfer edge vertex is out of range");
    validateIndex(edge[1], vertex_count, "geometry transfer edge vertex is out of range");
  }
  for (const auto& triangle : input.triangles) {
    validateIndex(triangle[0], vertex_count, "geometry transfer triangle vertex is out of range");
    validateIndex(triangle[1], vertex_count, "geometry transfer triangle vertex is out of range");
    validateIndex(triangle[2], vertex_count, "geometry transfer triangle vertex is out of range");
  }
  for (const auto& tet : input.tets) {
    validateIndex(tet[0], vertex_count, "geometry transfer tet vertex is out of range");
    validateIndex(tet[1], vertex_count, "geometry transfer tet vertex is out of range");
    validateIndex(tet[2], vertex_count, "geometry transfer tet vertex is out of range");
    validateIndex(tet[3], vertex_count, "geometry transfer tet vertex is out of range");
  }
#endif

  GeometryTransferMap map;
  map.sourceObject = input.sourceObject;
  map.points = GeometryRange{
      .first = geometry.pointCount(),
      .count = vertex_count,
  };
  map.localToGlobalPoint.resize(input.vertices.size());
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    const glm::dvec3& position = input.positions.empty()
                                     ? input.vertices[static_cast<size_t>(vertex)]
                                     : input.positions[static_cast<size_t>(vertex)];
    map.localToGlobalPoint[static_cast<size_t>(vertex)] =
        geometry.addPoint(input.subsystem,
                          input.localSampleOffset + vertex,
                          position,
                          input.vertices[static_cast<size_t>(vertex)],
                          input.radius);
  }

  map.edges = GeometryRange{
      .first = geometry.edgeCount(),
      .count = static_cast<int>(input.edges.size()),
  };
  map.localToGlobalEdge.reserve(input.edges.size());
  for (const auto& edge : input.edges) {
    map.localToGlobalEdge.push_back(
        geometry.addEdge(map.localToGlobalPoint[static_cast<size_t>(edge[0])],
                         map.localToGlobalPoint[static_cast<size_t>(edge[1])],
                         input.radius));
  }

  map.triangles = GeometryRange{
      .first = geometry.triangleCount(),
      .count = static_cast<int>(input.triangles.size()),
  };
  map.localToGlobalTriangle.reserve(input.triangles.size());
  for (const auto& triangle : input.triangles) {
    map.localToGlobalTriangle.push_back(geometry.addTriangle(
        map.localToGlobalPoint[static_cast<size_t>(triangle[0])],
        map.localToGlobalPoint[static_cast<size_t>(triangle[1])],
        map.localToGlobalPoint[static_cast<size_t>(triangle[2])],
        input.radius));
  }

  map.tets = GeometryRange{
      .first = geometry.tetCount(),
      .count = static_cast<int>(input.tets.size()),
  };
  map.localToGlobalTet.reserve(input.tets.size());
  for (const auto& tet : input.tets) {
    map.localToGlobalTet.push_back(geometry.addTet(
        map.localToGlobalPoint[static_cast<size_t>(tet[0])],
        map.localToGlobalPoint[static_cast<size_t>(tet[1])],
        map.localToGlobalPoint[static_cast<size_t>(tet[2])],
        map.localToGlobalPoint[static_cast<size_t>(tet[3])]));
  }

  return map;
}

}  // namespace ksk::runtime
