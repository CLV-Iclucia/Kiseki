#pragma once

#include <array>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include <Runtime/types.h>

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
  SubsystemId subsystem = -1;
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

struct GeometryBounds {
  glm::dvec3 lower{0.0};
  glm::dvec3 upper{0.0};
  bool empty = true;

  GeometryBounds& expand(const glm::dvec3& point) noexcept;
};

struct GeometryPoint {
  GeometryPointId id = -1;
  GeometryOwner owner;
  SubsystemId subsystem = -1;
  int localSampleId = -1;
  GeometryInstanceId instance = -1;
  int instanceVertex = -1;
  glm::dvec3 x{0.0};
};

struct GeometryEdge {
  int id = -1;
  GeometryReference ref;
  GeometryPointId p0 = -1;
  GeometryPointId p1 = -1;
  Real radius = 0.0;
};

struct GeometryTriangle {
  int id = -1;
  GeometryReference ref;
  GeometryPointId p0 = -1;
  GeometryPointId p1 = -1;
  GeometryPointId p2 = -1;
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
  bool crossesSubsystems = false;
  bool hasInstanceGeometry = false;
  int subsystemCount = 0;
  std::array<SubsystemId, 4> subsystems{};
};

struct GlobalGeometryManager {
  std::vector<GeometryPoint> points;
  std::vector<GeometryEdge> edges;
  std::vector<GeometryTriangle> triangles;
  std::vector<GeometryInstance> instances;

  [[nodiscard]] GeometryPointId appendPoint(SubsystemId subsystem,
                                            int localSampleId,
                                            const glm::dvec3& position);
  [[nodiscard]] GeometryPointId appendColliderPoint(
      int collider,
      int localSampleId,
      const glm::dvec3& position);
  [[nodiscard]] int appendEdge(GeometryPointId p0,
                               GeometryPointId p1,
                               Real radius = 0.0);
  [[nodiscard]] int appendTriangle(GeometryPointId p0,
                                   GeometryPointId p1,
                                   GeometryPointId p2);
  [[nodiscard]] GeometryInstanceId appendInstance(
      SubsystemId subsystem,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] GeometryInstanceId appendColliderInstance(
      int collider,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] bool contains(GeometryPointId point) const noexcept;

  [[nodiscard]] int pointCount() const noexcept;
  [[nodiscard]] int edgeCount() const noexcept;
  [[nodiscard]] int triangleCount() const noexcept;

  [[nodiscard]] GeometryRange pointRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange edgeRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange triangleRange(SubsystemId subsystem) const noexcept;
  [[nodiscard]] GeometryRange colliderPointRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderEdgeRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderTriangleRange(int collider) const noexcept;

  [[nodiscard]] GeometryReference pointRef(GeometryPointId point) const;
  [[nodiscard]] GeometryReference edgeRef(int edge) const;
  [[nodiscard]] GeometryReference triangleRef(int triangle) const;

  [[nodiscard]] GeometryOwner pointOwner(GeometryPointId point) const;
  [[nodiscard]] bool sameSubsystem(std::span<const GeometryPointId> stencil) const;
  [[nodiscard]] bool hasCollider(std::span<const GeometryPointId> stencil) const;
  [[nodiscard]] GeometryStencilInfo classify(
      std::span<const GeometryPointId> stencil) const;

  [[nodiscard]] GeometryPointId localToGlobalPoint(
      SubsystemId subsystem,
      int localSampleId) const;
  [[nodiscard]] GeometryPointId colliderLocalToGlobalPoint(
      int collider,
      int localSampleId) const;
  [[nodiscard]] std::array<GeometryPointId, 2> globalEdge(int edge) const;
  [[nodiscard]] std::array<GeometryPointId, 3> globalTriangle(
      int triangle) const;

  [[nodiscard]] bool triangleContainsPoint(int triangle,
                                           GeometryPointId point) const;
  [[nodiscard]] bool edgesAdjacent(int edgeA, int edgeB) const;

  [[nodiscard]] GeometryBounds pointBounds(GeometryPointId point) const;
  [[nodiscard]] GeometryBounds edgeBounds(int edge) const;
  [[nodiscard]] GeometryBounds triangleBounds(int triangle) const;
  [[nodiscard]] GeometryBounds trajectoryPointBounds(
      GeometryPointId point,
      std::span<const glm::dvec3> directions,
      Real toi) const;
  [[nodiscard]] GeometryBounds trajectoryEdgeBounds(
      int edge,
      std::span<const glm::dvec3> directions,
      Real toi) const;
  [[nodiscard]] GeometryBounds trajectoryTriangleBounds(
      int triangle,
      std::span<const glm::dvec3> directions,
      Real toi) const;

  [[nodiscard]] glm::dvec3 worldPosition(GeometryPointId point) const;
  void setPointPosition(GeometryPointId point, const glm::dvec3& position);
  void setInstanceTransform(GeometryInstanceId instance,
                            const glm::dmat4& transform);

 private:
  [[nodiscard]] GeometryRange rangeFor(
      const std::vector<GeometryRange>& ranges,
      int ownerIndex) const noexcept;
  [[nodiscard]] GeometryReference appendReference(
      std::vector<GeometryRange>& ranges,
      const GeometryOwner& owner,
      int nextGlobalIndex);
  [[nodiscard]] GeometryPointId appendPoint(
      const GeometryOwner& owner,
      int localSampleId,
      const glm::dvec3& position,
      GeometryInstanceId instance,
      int instanceVertex);
  [[nodiscard]] GeometryInstanceId appendInstance(
      const GeometryOwner& owner,
      int localInstanceId,
      const GeometryMeshDesc& mesh,
      const glm::dmat4& transform);
  [[nodiscard]] const GeometryPoint& checkedPoint(GeometryPointId point) const;
  [[nodiscard]] GeometryPoint& checkedPoint(GeometryPointId point);
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
