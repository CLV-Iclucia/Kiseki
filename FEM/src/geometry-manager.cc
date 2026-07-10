#include <fem/geometry-manager.h>

#include "fem/colliders.h"

namespace ksk::fem {

void GeometryManager::collectGeometryReferences(
    const std::vector<Primitive> &primitives,
    const std::vector<Collider> &colliders) {
  this->primitives = &primitives;

  triangleRefs.clear();
  edgeRefs.clear();
  vertexRefs.clear();

  primitiveTriangleRanges.resize(primitives.size());
  primitiveEdgeRanges.resize(primitives.size());
  primitiveVertexRanges.resize(primitives.size());

  int vertexOffset = 0;

  for (int primId = 0; primId < primitives.size(); ++primId) {
    const auto &primitive = primitives[primId];

    int triangleStart = static_cast<int>(triangleRefs.size());
    std::span<const Triangle> surfaceTriangles = primitive.getSurfaceView();
    for (int localIdx = 0; localIdx < surfaceTriangles.size(); ++localIdx)
      triangleRefs.emplace_back(primId, localIdx);
    primitiveTriangleRanges[primId] = {triangleStart,
                                       static_cast<int>(triangleRefs.size())};

    int edgeStart = static_cast<int>(edgeRefs.size());
    std::span<const Edge> edges = primitive.getEdgesView();
    for (int localIdx = 0; localIdx < edges.size(); ++localIdx)
      edgeRefs.emplace_back(primId, localIdx);
    primitiveEdgeRanges[primId] = {edgeStart,
                                   static_cast<int>(edgeRefs.size())};

    int vertexStart = static_cast<int>(vertexRefs.size());
    int vertexCount = static_cast<int>(primitive.getVertexCount());

    for (int localIdx = 0; localIdx < vertexCount; ++localIdx)
      vertexRefs.emplace_back(primId, localIdx);

    primitiveVertexRanges[primId] = {vertexStart,
                                     static_cast<int>(vertexRefs.size())};
    vertexOffset += vertexCount;
  }

  // TODO: process collider
}

int GeometryManager::localToGlobalVertex(int primitiveId,
                                         int localVertexIdx) const {
  const auto &range = primitiveVertexRanges[primitiveId];
  return localVertexIdx + range.first;
}

Triangle GeometryManager::getTriangle(int globalIndex) const {
  auto ref = triangleRefs[globalIndex];
  return (*primitives)[ref.primitiveId].getSurfaceView()[ref.localIndex];
}

Edge GeometryManager::getEdge(int globalIndex) const {
  auto ref = edgeRefs[globalIndex];
  return (*primitives)[ref.primitiveId].getEdgesView()[ref.localIndex];
}

Triangle GeometryManager::getGlobalTriangle(int globalIndex) const {
  Triangle tri = getTriangle(globalIndex);
  auto ref = triangleRefs[globalIndex];
  return tri + primitiveVertexRanges[ref.primitiveId].first;
}

Edge GeometryManager::getGlobalEdge(int globalIndex) const {
  const Edge &edge = getEdge(globalIndex);
  const auto &ref = edgeRefs[globalIndex];
  const auto &range = primitiveVertexRanges[ref.primitiveId];
  return edge + range.first;
}

bool GeometryManager::triangleContainsVertex(int triangleIdx,
                                             int vertexIdx) const {
  const auto vertices = getGlobalTriangle(triangleIdx);
  return vertices.x == vertexIdx || vertices.y == vertexIdx ||
         vertices.z == vertexIdx;
}

bool GeometryManager::checkEdgeAdjacent(int edgeA, int edgeB) const {
  const auto verticesA = getGlobalEdge(edgeA);
  const auto verticesB = getGlobalEdge(edgeB);

  return verticesA.x == verticesB.x || verticesA.x == verticesB.y ||
         verticesA.y == verticesB.x || verticesA.y == verticesB.y;
}

BBox<Real, 3> GeometryManager::TriangleAccessor::bbox(int idx) const {
  const auto vertices = manager.getGlobalTriangle(idx);

  BBox<Real, 3> box;
  std::array<int, 3> vertexIds = {vertices.x, vertices.y, vertices.z};
  for (int i = 0; i < 3; ++i) {
    const auto& pos = positions[vertexIds[i]];
    box.expand({pos.x, pos.y, pos.z});
  }
  return box;
}

BBox<Real, 3> GeometryManager::EdgeAccessor::bbox(int idx) const {
  auto vertices = manager.getGlobalEdge(idx);

  BBox<Real, 3> box;
  std::array<int, 2> vertexIds = {vertices.x, vertices.y};
  for (int i = 0; i < 2; ++i) {
    const auto& pos = positions[vertexIds[i]];
    box.expand({pos.x, pos.y, pos.z});
  }
  return box;
}

BBox<Real, 3> GeometryManager::VertexAccessor::bbox(int idx) const {
  const auto& pos = positions[idx];
  return BBox<Real, 3>({pos.x, pos.y, pos.z});
}

BBox<Real, 3> GeometryManager::TrajectoryAccessor::triangleBBox(int idx) const {
  const auto vertices = manager.getGlobalTriangle(idx);

  BBox<Real, 3> box;
  std::array<int, 3> vertexIds = {vertices.x, vertices.y, vertices.z};
  for (int i = 0; i < 3; ++i) {
    int vIdx = vertexIds[i];
    const auto& startPos = positions[vIdx];
    const auto& dir = directions[vIdx];
    auto endPos = startPos + dir * toi;

    box.expand({startPos.x, startPos.y, startPos.z});
    box.expand({endPos.x, endPos.y, endPos.z});
  }
  return box;
}

BBox<Real, 3> GeometryManager::TrajectoryAccessor::edgeBBox(int idx) const {
  const auto vertices = manager.getGlobalEdge(idx);

  BBox<Real, 3> box;
  std::array<int, 2> vertexIds = {vertices.x, vertices.y};
  for (int i = 0; i < 2; ++i) {
    int vIdx = vertexIds[i];
    const auto& startPos = positions[vIdx];
    const auto& dir = directions[vIdx];
    auto endPos = startPos + dir * toi;

    box.expand({startPos.x, startPos.y, startPos.z});
    box.expand({endPos.x, endPos.y, endPos.z});
  }
  return box;
}

BBox<Real, 3> GeometryManager::TrajectoryAccessor::vertexBBox(int idx) const {
  const auto& startPos = positions[idx];
  auto endPos = startPos + directions[idx] * toi;

  return BBox<Real, 3>({startPos.x, startPos.y, startPos.z})
      .expand({endPos.x, endPos.y, endPos.z});
}

} // namespace ksk::fem