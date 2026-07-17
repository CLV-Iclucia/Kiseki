#include <Runtime/global-geometry-manager.h>

#include <stdexcept>

namespace ksk::runtime {

PointIdx GlobalGeometryManager::addPoint(SubsystemId subsystem,
                                           int localSampleId,
                                           const glm::dvec3& position)
{
  return addPoint(subsystem, localSampleId, position, position);
}

PointIdx GlobalGeometryManager::addPoint(
    SubsystemId subsystem,
    int localSampleId,
    const glm::dvec3& position,
    const glm::dvec3& restPosition)
{
  return addPoint(GeometryOwner{
                         .subsystem = subsystem,
                     },
                     localSampleId,
                     position,
                     restPosition,
                     -1,
                     -1);
}

PointIdx GlobalGeometryManager::addColliderPoint(
    int collider,
    int localSampleId,
    const glm::dvec3& position)
{
  return addColliderPoint(collider, localSampleId, position, position);
}

PointIdx GlobalGeometryManager::addColliderPoint(
    int collider,
    int localSampleId,
    const glm::dvec3& position,
    const glm::dvec3& restPosition)
{
  return addPoint(GeometryOwner{
                         .collider = collider,
                     },
                     localSampleId,
                     position,
                     restPosition,
                     -1,
                     -1);
}

PointIdx GlobalGeometryManager::addPoint(
    const GeometryOwner& owner,
    int localSampleId,
    const glm::dvec3& position,
    const glm::dvec3& restPosition,
    GeometryInstanceId instance,
    int instanceVertex)
{
  PointIdx id = static_cast<int>(points.size());
  if (owner.isSubsystem()) {
    addReference(point_ranges_, owner, id);
  } else if (owner.isCollider()) {
    addReference(collider_point_ranges_, owner, id);
  } else {
    throw std::invalid_argument("geometry point owner is invalid");
  }

  points.push_back(GeometryPoint{
      .owner = owner,
      .localSampleId = localSampleId,
      .instance = instance,
      .instanceVertex = instanceVertex,
      .x = position,
      .restX = restPosition,
  });
  return id;
}

int GlobalGeometryManager::addEdge(PointIdx p0, PointIdx p1, Real radius)
{
  const GeometryPoint& first = checkedPoint(p0);
  const GeometryPoint& second = checkedPoint(p1);
  if (first.owner.subsystem != second.owner.subsystem ||
      first.owner.collider != second.owner.collider) {
    throw std::invalid_argument(
        "geometry edge endpoints belong to different owners");
  }
  const int id = static_cast<int>(edges.size());
  const GeometryReference ref = first.owner.isSubsystem()
                                    ? addReference(edge_ranges_,
                                                      first.owner,
                                                      id)
                                    : addReference(collider_edge_ranges_,
                                                      first.owner,
                                                      id);
  edges.push_back(GeometryEdge{
      .id = id,
      .ref = ref,
      .p0 = p0,
      .p1 = p1,
      .radius = radius,
  });
  return id;
}

int GlobalGeometryManager::addTriangle(PointIdx p0,
                                          PointIdx p1,
                                          PointIdx p2,
                                          Real radius)
{
  const GeometryPoint& first = checkedPoint(p0);
  const GeometryPoint& second = checkedPoint(p1);
  const GeometryPoint& third = checkedPoint(p2);
  if (first.owner.subsystem != second.owner.subsystem ||
      first.owner.subsystem != third.owner.subsystem ||
      first.owner.collider != second.owner.collider ||
      first.owner.collider != third.owner.collider) {
    throw std::invalid_argument(
        "geometry triangle vertices belong to different owners");
  }
  const int id = static_cast<int>(triangles.size());
  const GeometryReference ref = first.owner.isSubsystem()
                                    ? addReference(triangle_ranges_,
                                                      first.owner,
                                                      id)
                                    : addReference(
                                          collider_triangle_ranges_,
                                          first.owner,
                                          id);
  triangles.push_back(GeometryTriangle{
      .id = id,
      .ref = ref,
      .p0 = p0,
      .p1 = p1,
      .p2 = p2,
      .thickness = radius,
  });
  return id;
}

GeometryInstanceId GlobalGeometryManager::addInstance(
    SubsystemId subsystem,
    int localInstanceId,
    const GeometryMeshDesc& mesh,
    const glm::dmat4& transform)
{
  return addInstance(GeometryOwner{
                            .subsystem = subsystem,
                        },
                        localInstanceId,
                        mesh,
                        transform);
}

GeometryInstanceId GlobalGeometryManager::addColliderInstance(
    int collider,
    int localInstanceId,
    const GeometryMeshDesc& mesh,
    const glm::dmat4& transform)
{
  return addInstance(GeometryOwner{
                            .collider = collider,
                        },
                        localInstanceId,
                        mesh,
                        transform);
}

GeometryInstanceId GlobalGeometryManager::addInstance(
    const GeometryOwner& owner,
    int localInstanceId,
    const GeometryMeshDesc& mesh,
    const glm::dmat4& transform)
{
  if (mesh.vertices.empty()) {
    throw std::invalid_argument("geometry instance mesh has no vertices");
  }
  const int vertex_count = static_cast<int>(mesh.vertices.size());
  for (const auto& edge : mesh.edges) {
    if (edge[0] < 0 || edge[0] >= vertex_count ||
        edge[1] < 0 || edge[1] >= vertex_count) {
      throw std::out_of_range("geometry instance edge vertex is out of range");
    }
  }
  for (const auto& triangle : mesh.triangles) {
    if (triangle[0] < 0 || triangle[0] >= vertex_count ||
        triangle[1] < 0 || triangle[1] >= vertex_count ||
        triangle[2] < 0 || triangle[2] >= vertex_count) {
      throw std::out_of_range(
          "geometry instance triangle vertex is out of range");
    }
  }

  const GeometryInstanceId instance_id = static_cast<int>(instances.size());
  const GeometryRange point_range{
      .first = static_cast<int>(points.size()),
      .count = static_cast<int>(mesh.vertices.size()),
  };
  for (int vertex = 0; vertex < static_cast<int>(mesh.vertices.size());
       ++vertex) {
    addPoint(owner,
                vertex,
                mesh.vertices[static_cast<size_t>(vertex)],
                mesh.vertices[static_cast<size_t>(vertex)],
                instance_id,
                vertex);
  }

  const GeometryRange edge_range{
      .first = static_cast<int>(edges.size()),
      .count = static_cast<int>(mesh.edges.size()),
  };
  for (const auto& edge : mesh.edges) {
    addEdge(PointIdx{point_range.first + edge[0]},
               PointIdx{point_range.first + edge[1]},
               mesh.radius);
  }

  const GeometryRange triangle_range{
      .first = static_cast<int>(triangles.size()),
      .count = static_cast<int>(mesh.triangles.size()),
  };
  for (const auto& triangle : mesh.triangles) {
    addTriangle(PointIdx{point_range.first + triangle[0]},
                   PointIdx{point_range.first + triangle[1]},
                   PointIdx{point_range.first + triangle[2]},
                   mesh.radius);
  }

  instances.push_back(GeometryInstance{
      .id = instance_id,
      .owner = owner,
      .localInstanceId = localInstanceId,
      .points = point_range,
      .edges = edge_range,
      .triangles = triangle_range,
      .transform = transform,
  });
  return instance_id;
}

bool GlobalGeometryManager::contains(PointIdx point) const noexcept
{
  return point >= 0 && point < static_cast<int>(points.size());
}

int GlobalGeometryManager::pointCount() const noexcept
{
  return static_cast<int>(points.size());
}

int GlobalGeometryManager::edgeCount() const noexcept
{
  return static_cast<int>(edges.size());
}

int GlobalGeometryManager::triangleCount() const noexcept
{
  return static_cast<int>(triangles.size());
}

GeometryRange GlobalGeometryManager::pointRange(SubsystemId subsystem) const noexcept
{
  return rangeFor(point_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::edgeRange(SubsystemId subsystem) const noexcept
{
  return rangeFor(edge_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::triangleRange(SubsystemId subsystem) const noexcept
{
  return rangeFor(triangle_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::colliderPointRange(int collider) const noexcept
{
  return rangeFor(collider_point_ranges_, collider);
}

GeometryRange GlobalGeometryManager::colliderEdgeRange(int collider) const noexcept
{
  return rangeFor(collider_edge_ranges_, collider);
}

GeometryRange GlobalGeometryManager::colliderTriangleRange(int collider) const noexcept
{
  return rangeFor(collider_triangle_ranges_, collider);
}

GeometryReference GlobalGeometryManager::pointRef(PointIdx point) const
{
  const GeometryPoint& entry = checkedPoint(point);
  return {
      .owner = entry.owner,
      .localIndex = entry.localSampleId,
  };
}

GeometryReference GlobalGeometryManager::edgeRef(int edge) const
{
  return checkedEdge(edge).ref;
}

GeometryReference GlobalGeometryManager::triangleRef(int triangle) const
{
  return checkedTriangle(triangle).ref;
}

GeometryOwner GlobalGeometryManager::pointOwner(PointIdx point) const
{
  return checkedPoint(point).owner;
}

bool GlobalGeometryManager::sameSubsystem(std::span<const PointIdx> stencil) const
{
  const GeometryStencilInfo info = classify(stencil);
  return info.valid && !info.hasCollider && info.subsystemCount == 1;
}

bool GlobalGeometryManager::hasCollider(std::span<const PointIdx> stencil) const
{
  return classify(stencil).hasCollider;
}

GeometryStencilInfo GlobalGeometryManager::classify(std::span<const PointIdx> stencil) const
{
  GeometryStencilInfo info;
  for (const PointIdx point : stencil) {
    const GeometryPoint& entry = checkedPoint(point);
    info.hasInstanceGeometry = info.hasInstanceGeometry || entry.instance >= 0;
    if (entry.owner.isCollider()) {
      info.hasCollider = true;
      continue;
    }
    if (!entry.owner.isSubsystem()) {
      info.valid = false;
      continue;
    }

    const SubsystemId subsystem = entry.owner.subsystem;
    if (info.subsystemCount == 0) {
      info.subsystems[0] = subsystem;
      info.subsystemCount = 1;
      continue;
    }
    if (info.subsystems[0] == subsystem) {
      continue;
    }
    if (info.subsystemCount == 1) {
      info.subsystems[1] = subsystem;
      info.subsystemCount = 2;
      continue;
    }
    if (info.subsystems[1] != subsystem) {
      info.valid = false;
    }
  }
  return info;
}

PointIdx GlobalGeometryManager::localToGlobalPoint(SubsystemId subsystem,
                                                  int localSampleId) const
{
  const GeometryRange range = pointRange(subsystem);
  for (int point = range.first; point < range.end(); ++point) {
    if (points[point].localSampleId == localSampleId) {
      return point;
    }
  }
  return -1;
}

PointIdx GlobalGeometryManager::colliderLocalToGlobalPoint(
    int collider,
    int localSampleId) const
{
  const GeometryRange range = colliderPointRange(collider);
  for (int point = range.first; point < range.end(); ++point) {
    if (points[static_cast<size_t>(point)].localSampleId == localSampleId) {
      return point;
    }
  }
  return -1;
}

std::array<PointIdx, 2> GlobalGeometryManager::globalEdge(int edge) const
{
  const GeometryEdge& entry = checkedEdge(edge);
  return {entry.p0, entry.p1};
}

std::array<PointIdx, 3> GlobalGeometryManager::globalTriangle(int triangle) const
{
  const GeometryTriangle& entry = checkedTriangle(triangle);
  return {entry.p0, entry.p1, entry.p2};
}

bool GlobalGeometryManager::triangleContainsPoint(int triangle,
                                          PointIdx point) const
{
  const auto vertices = globalTriangle(triangle);
  return vertices[0] == point || vertices[1] == point || vertices[2] == point;
}

bool GlobalGeometryManager::edgesAdjacent(int edgeA, int edgeB) const
{
  const auto a = globalEdge(edgeA);
  const auto b = globalEdge(edgeB);
  return a[0] == b[0] || a[0] == b[1] ||
         a[1] == b[0] || a[1] == b[1];
}

GeometryBounds GlobalGeometryManager::pointBounds(PointIdx point) const
{
  GeometryBounds bounds;
  bounds.expand(worldPosition(point));
  return bounds;
}

GeometryBounds GlobalGeometryManager::edgeBounds(int edge) const
{
  GeometryBounds bounds;
  const auto vertices = globalEdge(edge);
  bounds.expand(worldPosition(vertices[0]));
  bounds.expand(worldPosition(vertices[1]));
  return bounds;
}

GeometryBounds GlobalGeometryManager::triangleBounds(int triangle) const
{
  GeometryBounds bounds;
  const auto vertices = globalTriangle(triangle);
  for (const PointIdx point : vertices) {
    bounds.expand(worldPosition(point));
  }
  return bounds;
}

GeometryBounds GlobalGeometryManager::trajectoryPointBounds(
    PointIdx point,
    std::span<const glm::dvec3> directions,
    Real toi) const
{
  checkedPoint(point);
  const glm::dvec3 position = worldPosition(point);
  GeometryBounds bounds;
  bounds.expand(position);
  bounds.expand(position + directions[point] * toi);
  return bounds;
}

GeometryBounds GlobalGeometryManager::trajectoryEdgeBounds(
    int edge,
    std::span<const glm::dvec3> directions,
    Real toi, Real radius) const
{
  GeometryBounds bounds;
  const auto vertices = globalEdge(edge);
  for (const PointIdx point : vertices) {
    const GeometryBounds point_bounds = trajectoryPointBounds(point, directions, toi);
    bounds.expand(point_bounds.lo);
    bounds.expand(point_bounds.hi);
  }
  return bounds;
}

glm::dvec3 GlobalGeometryManager::worldPosition(PointIdx point) const
{
  const GeometryPoint& entry = checkedPoint(point);
  if (entry.instance < 0) {
    return entry.x;
  }
  const GeometryInstance& instance = checkedInstance(entry.instance);
  return glm::dvec3(instance.transform * glm::dvec4(entry.x, 1.0));
}

glm::dvec3 GlobalGeometryManager::restPosition(PointIdx point) const
{
  const GeometryPoint& entry = checkedPoint(point);
  if (entry.instance < 0) {
    return entry.restX;
  }
  const GeometryInstance& instance = checkedInstance(entry.instance);
  return glm::dvec3(instance.transform * glm::dvec4(entry.restX, 1.0));
}

void GlobalGeometryManager::setPointPosition(PointIdx point,
                                     const glm::dvec3& position)
{
  GeometryPoint& entry = checkedPoint(point);
  if (entry.instance >= 0) {
    throw std::invalid_argument(
        "cannot directly set an instance vertex position");
  }
  entry.x = position;
}

void GlobalGeometryManager::setPointRestPosition(
    PointIdx point,
    const glm::dvec3& restPosition)
{
  GeometryPoint& entry = checkedPoint(point);
  if (entry.instance >= 0) {
    throw std::invalid_argument(
        "cannot directly set an instance vertex rest position");
  }
  entry.restX = restPosition;
}

void GlobalGeometryManager::setInstanceTransform(GeometryInstanceId instance,
                                         const glm::dmat4& transform)
{
  checkedInstance(instance).transform = transform;
}

spatify::BBox<Real, 3> GlobalGeometryManager::trajectoryTriangleBounds(
    int triangle,
    std::span<const glm::dvec3> directions,
    Real toi, Real thickness) const
{
  GeometryBounds bounds;
  const auto vertices = globalTriangle(triangle);
  for (const PointIdx point : vertices) {
    const GeometryBounds point_bounds = trajectoryPointBounds(point, directions, toi);
    bounds.expand(point_bounds.lo);
    bounds.expand(point_bounds.hi);
  }
  return bounds;
}

GeometryRange GlobalGeometryManager::rangeFor(
    const std::vector<GeometryRange>& ranges,
    int ownerIndex) const noexcept
{
  if (ownerIndex < 0 || ownerIndex >= static_cast<int>(ranges.size())) {
    return {};
  }
  return ranges[static_cast<size_t>(ownerIndex)];
}

GeometryReference GlobalGeometryManager::addReference(
    std::vector<GeometryRange>& ranges,
    const GeometryOwner& owner,
    int nextGlobalIndex)
{
  int owner_index = -1;
  if (owner.isSubsystem()) {
    owner_index = owner.subsystem;
  } else if (owner.isCollider()) {
    owner_index = owner.collider;
  } else {
    throw std::invalid_argument("geometry owner is invalid");
  }
  if (ranges.size() <= static_cast<size_t>(owner_index)) {
    ranges.resize(static_cast<size_t>(owner_index + 1));
  }

  GeometryRange& range = ranges[static_cast<size_t>(owner_index)];
  if (range.count == 0) {
    range.first = nextGlobalIndex;
  } else if (nextGlobalIndex != range.end()) {
    throw std::invalid_argument(
        "geometry for a subsystem must be appended contiguously");
  }

  const GeometryReference ref{
      .owner = owner,
      .localIndex = range.count,
  };
  ++range.count;
  return ref;
}

const GeometryPoint& GlobalGeometryManager::checkedPoint(PointIdx point) const
{
  assert(contains(point));
  return points[point];
}

GeometryPoint& GlobalGeometryManager::checkedPoint(PointIdx point)
{
  assert(contains(point));
  return points[point];
}

const GeometryEdge& GlobalGeometryManager::checkedEdge(int edge) const
{
  assert(edge >= 0 && edge < static_cast<int>(edges.size()));
  return edges[edge];
}

const GeometryTriangle& GlobalGeometryManager::checkedTriangle(int triangle) const
{
  assert(triangle >= 0 && triangle < triangles.size());
  return triangles[triangle];
}

const GeometryInstance& GlobalGeometryManager::checkedInstance(
    GeometryInstanceId instance) const
{
  if (instance < 0 || instance >= static_cast<int>(instances.size())) {
    throw std::out_of_range("geometry instance id is out of range");
  }
  return instances[static_cast<size_t>(instance)];
}

GeometryInstance& GlobalGeometryManager::checkedInstance(GeometryInstanceId instance)
{
  if (instance < 0 || instance >= static_cast<int>(instances.size())) {
    throw std::out_of_range("geometry instance id is out of range");
  }
  return instances[static_cast<size_t>(instance)];
}

}  // namespace ksk::runtime
