#pragma once

#include <array>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include <Runtime/types.h>

#include <Spatify/bbox.h>

namespace ksk::runtime {

using GeometryInstanceId = int;

struct GeometryOwner {
  SubsystemId subsystem = -1;
  int collider = -1;

  [[nodiscard]] bool isSubsystem() const noexcept
  {
    return subsystem >= 0 && collider < 0;
  }
  [[nodiscard]] bool isCollider() const noexcept
  {
    return subsystem < 0 && collider >= 0;
  }
};

struct GeometryReference {
  GeometryOwner owner;
  int localIndex = -1;
};

struct GeometryRange {
  int first = 0;
  int count = 0;

  [[nodiscard]] int end() const noexcept { return first + count; }
  [[nodiscard]] bool contains(int index) const noexcept
  {
    return index >= first && index < end();
  }
};

using GeometryBounds = spatify::BBox<Real, 3>;
struct GeometryPoint {
  GeometryOwner owner;
  int localSampleId = -1;
  GeometryInstanceId instance = -1;
  int instanceVertex = -1;
  Real radius = 0.0;
  glm::dvec3 x{0.0};
  glm::dvec3 restX{0.0};
};

struct GeometryEdge {
  int id = -1;
  GeometryReference ref;
  PointIdx p0 = -1;
  PointIdx p1 = -1;
  Real radius = 0.0;
};

struct GeometryTriangle {
  int id = -1;
  GeometryReference ref;
  PointIdx p0 = -1;
  PointIdx p1 = -1;
  PointIdx p2 = -1;
  Real thickness = 0.0;
};

struct GeometryMeshDesc {
  std::vector<glm::dvec3> vertices;
  std::vector<std::array<int, 2>> edges;
  std::vector<std::array<int, 3>> triangles;
  Real radius = 0.0;
};

struct GeometryInstance {
  GeometryInstanceId id = -1;
  GeometryOwner owner;
  int localInstanceId = -1;
  GeometryRange points;
  GeometryRange edges;
  GeometryRange triangles;
  glm::dmat4 transform{1.0};
};

struct GeometryStencilInfo {
  bool valid = true;
  bool hasCollider = false;
  bool hasInstanceGeometry = false;
  int subsystemCount = 0;
  std::array<SubsystemId, 2> subsystems{};
};

struct GlobalGeometryManager {
  std::vector<GeometryPoint> points;
  std::vector<GeometryEdge> edges;
  std::vector<GeometryTriangle> triangles;
  std::vector<GeometryInstance> instances;

  [[nodiscard]] PointIdx addPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position);
  [[nodiscard]] PointIdx addPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position,
                                            Real radius);
  [[nodiscard]] PointIdx addPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position,
                                            const glm::dvec3& restPosition);
  [[nodiscard]] PointIdx addPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position,
                                            const glm::dvec3& restPosition,
                                            Real radius);
  [[nodiscard]] PointIdx addColliderPoint(
      int collider,
      int localSampleId,
      const glm::dvec3& position);
  [[nodiscard]] PointIdx addColliderPoint(
      int collider,
      int localSampleId,
      const glm::dvec3& position,
      Real radius);
  [[nodiscard]] PointIdx addColliderPoint(
      int collider,
      int localSampleId,
      const glm::dvec3& position,
      const glm::dvec3& restPosition);
  [[nodiscard]] PointIdx addColliderPoint(
      int collider,
      int localSampleId,
      const glm::dvec3& position,
      const glm::dvec3& restPosition,
      Real radius);
  [[nodiscard]] int addEdge(PointIdx p0,
                               PointIdx p1,
                               Real radius = 0.0);
  [[nodiscard]] int addTriangle(PointIdx p0, PointIdx p1, PointIdx p2, Real thickness = 0.0);
  [[nodiscard]] GeometryInstanceId addInstance(
      SubsystemId subsystem,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] GeometryInstanceId addColliderInstance(
      int collider,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] bool contains(PointIdx point) const noexcept;

  [[nodiscard]] int pointCount() const noexcept;
  [[nodiscard]] int edgeCount() const noexcept;
  [[nodiscard]] int triangleCount() const noexcept;

  [[nodiscard]] GeometryRange pointRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange edgeRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange triangleRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange colliderPointRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderEdgeRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderTriangleRange(int collider) const noexcept;

  [[nodiscard]] GeometryReference pointRef(PointIdx point) const;
  [[nodiscard]] GeometryReference edgeRef(int edge) const;
  [[nodiscard]] GeometryReference triangleRef(int triangle) const;

  [[nodiscard]] GeometryOwner pointOwner(PointIdx point) const;
  [[nodiscard]] bool sameSubsystem(std::span<const PointIdx> stencil) const;
  [[nodiscard]] bool hasCollider(std::span<const PointIdx> stencil) const;
  [[nodiscard]] GeometryStencilInfo classify(std::span<const PointIdx> stencil) const;

  [[nodiscard]] PointIdx localToGlobalPoint(
      SubsystemId subsystem,
      int localSampleId) const;
  [[nodiscard]] PointIdx colliderLocalToGlobalPoint(
      int collider,
      int localSampleId) const;
  [[nodiscard]] std::array<PointIdx, 2> globalEdge(int edge) const;
  [[nodiscard]] std::array<PointIdx, 3> globalTriangle(int triangle) const;

  [[nodiscard]] bool triangleContainsPoint(int triangle,
                                           PointIdx point) const;
  [[nodiscard]] bool edgesAdjacent(int edgeA, int edgeB) const;

  [[nodiscard]] GeometryBounds pointBounds(PointIdx point) const;
  [[nodiscard]] GeometryBounds edgeBounds(int edge) const;
  [[nodiscard]] GeometryBounds triangleBounds(int triangle) const;
  [[nodiscard]] GeometryBounds trajectoryPointBounds(
      PointIdx point,
      std::span<const glm::dvec3> directions,
      Real toi) const;
  [[nodiscard]] GeometryBounds trajectoryEdgeBounds(
      int edge,
      std::span<const glm::dvec3> directions,
      Real toi, Real radius = 0.0) const;
  [[nodiscard]] GeometryBounds trajectoryTriangleBounds(
      int triangle,
      std::span<const glm::dvec3> directions,
      Real toi, Real thickness = 0.0) const;

  [[nodiscard]] glm::dvec3 worldPosition(PointIdx point) const;
  [[nodiscard]] glm::dvec3 restPosition(PointIdx point) const;
  void setPointPosition(PointIdx point, const glm::dvec3& position);
  void setPointRestPosition(PointIdx point,
                            const glm::dvec3& restPosition);
  void setInstanceTransform(GeometryInstanceId instance,
                            const glm::dmat4& transform);

 private:
  [[nodiscard]] GeometryRange rangeFor(
      const std::vector<GeometryRange>& ranges,
      int ownerIndex) const noexcept;
  [[nodiscard]] GeometryReference addReference(
      std::vector<GeometryRange>& ranges,
      const GeometryOwner& owner,
      int nextGlobalIndex);
  [[nodiscard]] PointIdx addPoint(
      const GeometryOwner& owner,
      int localSampleId,
      const glm::dvec3& position,
      const glm::dvec3& restPosition,
      Real radius,
      GeometryInstanceId instance,
      int instanceVertex);
  [[nodiscard]] GeometryInstanceId addInstance(
      const GeometryOwner& owner,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform);
  [[nodiscard]] const GeometryPoint& checkedPoint(PointIdx point) const;
  [[nodiscard]] GeometryPoint& checkedPoint(PointIdx point);
  [[nodiscard]] const GeometryEdge& checkedEdge(int edge) const;
  [[nodiscard]] const GeometryTriangle& checkedTriangle(int triangle) const;
  [[nodiscard]] const GeometryInstance& checkedInstance(
      GeometryInstanceId instance) const;
  [[nodiscard]] GeometryInstance& checkedInstance(GeometryInstanceId instance);

  std::vector<GeometryRange> point_ranges_;
  std::vector<GeometryRange> edge_ranges_;
  std::vector<GeometryRange> triangle_ranges_;
  std::vector<GeometryRange> collider_point_ranges_;
  std::vector<GeometryRange> collider_edge_ranges_;
  std::vector<GeometryRange> collider_triangle_ranges_;
};

}  // namespace ksk::runtime
