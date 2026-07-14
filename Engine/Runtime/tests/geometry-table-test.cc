#include <Runtime/global-geometry-manager.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace ksk::runtime {
namespace {

TEST(GeometryTable, PreservesPointOwnership)
{
  GlobalGeometryManager geometry;

  const auto p0 =
      geometry.appendPoint(SubsystemId{4}, 10, glm::dvec3{1.0, 0.0, 0.0});
  const auto p1 =
      geometry.appendPoint(SubsystemId{4}, 11, glm::dvec3{2.0, 0.0, 0.0});

  EXPECT_TRUE(geometry.contains(p0));
  EXPECT_TRUE(geometry.contains(p1));
  EXPECT_EQ(geometry.points[static_cast<size_t>(p0)].subsystem,
            SubsystemId{4});
  EXPECT_EQ(geometry.points[static_cast<size_t>(p1)].localSampleId, 11);
}

TEST(GeometryTable, AppendsEdgesAndTrianglesWithStableIds)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.appendPoint(SubsystemId{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 =
      geometry.appendPoint(SubsystemId{0}, 2, glm::dvec3{0.0, 1.0, 0.0});

  const int edge_id = geometry.appendEdge(p0, p1, 0.05);
  const int triangle_id = geometry.appendTriangle(p0, p1, p2);

  EXPECT_EQ(edge_id, 0);
  EXPECT_EQ(triangle_id, 0);
  EXPECT_EQ(geometry.edges[edge_id].p0, p0);
  EXPECT_DOUBLE_EQ(geometry.edges[edge_id].radius, 0.05);
  EXPECT_EQ(geometry.triangles[triangle_id].p2, p2);
  EXPECT_EQ(geometry.edgeRef(edge_id).subsystem, SubsystemId{0});
  EXPECT_EQ(geometry.triangleRef(triangle_id).localIndex, 0);
}

TEST(GeometryTable, TracksSubsystemRangesAndLocalPointMapping)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{0}, 7, glm::dvec3{0.0});
  const auto p1 = geometry.appendPoint(SubsystemId{0}, 8, glm::dvec3{1.0});
  const auto p2 = geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3{2.0});

  EXPECT_EQ(geometry.pointCount(), 3);
  EXPECT_EQ(geometry.pointRange(SubsystemId{0}).first, 0);
  EXPECT_EQ(geometry.pointRange(SubsystemId{0}).count, 2);
  EXPECT_EQ(geometry.pointRange(SubsystemId{1}).first, 2);
  EXPECT_EQ(geometry.pointRange(SubsystemId{1}).count, 1);
  EXPECT_EQ(geometry.localToGlobalPoint(SubsystemId{0}, 8), p1);
  EXPECT_EQ(geometry.localToGlobalPoint(SubsystemId{1}, 0), p2);
  EXPECT_LT(geometry.localToGlobalPoint(SubsystemId{1}, 99), 0);
  EXPECT_EQ(geometry.pointRef(p0).localIndex, 7);
}

TEST(GeometryTable, AnswersTopologyQueries)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{0.0});
  const auto p1 = geometry.appendPoint(SubsystemId{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 = geometry.appendPoint(SubsystemId{0}, 2, glm::dvec3{0.0, 1.0, 0.0});
  const int e0 = geometry.appendEdge(p0, p1);
  const int e1 = geometry.appendEdge(p1, p2);
  const int tri = geometry.appendTriangle(p0, p1, p2);

  EXPECT_TRUE(geometry.edgesAdjacent(e0, e1));
  EXPECT_TRUE(geometry.triangleContainsPoint(tri, p2));
  EXPECT_FALSE(geometry.triangleContainsPoint(tri, GeometryPointId{99}));

  const auto edge_vertices = geometry.globalEdge(e0);
  EXPECT_EQ(edge_vertices[0], p0);
  EXPECT_EQ(edge_vertices[1], p1);
}

TEST(GeometryTable, ClassifiesConstraintOwnership)
{
  GlobalGeometryManager geometry;

  const auto a0 = geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{0.0});
  const auto a1 = geometry.appendPoint(SubsystemId{0}, 1, glm::dvec3{1.0});
  const auto b0 = geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3{2.0});
  const auto c0 = geometry.appendColliderPoint(0, 0, glm::dvec3{3.0});

  const std::array same{a0, a1};
  const GeometryStencilInfo same_info = geometry.classify(same);
  EXPECT_TRUE(geometry.sameSubsystem(same));
  EXPECT_FALSE(same_info.crossesSubsystems);
  EXPECT_EQ(same_info.subsystemCount, 1);
  EXPECT_EQ(same_info.subsystems[0], SubsystemId{0});

  const std::array cross{a0, b0};
  const GeometryStencilInfo cross_info = geometry.classify(cross);
  EXPECT_FALSE(geometry.sameSubsystem(cross));
  EXPECT_TRUE(cross_info.crossesSubsystems);
  EXPECT_EQ(cross_info.subsystemCount, 2);

  const std::array collider{a0, c0};
  const GeometryStencilInfo collider_info = geometry.classify(collider);
  EXPECT_TRUE(collider_info.hasCollider);
  EXPECT_TRUE(geometry.hasCollider(collider));
}

TEST(GeometryTable, ComputesStaticAndTrajectoryBounds)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.appendPoint(SubsystemId{0}, 1, glm::dvec3{1.0, 2.0, 3.0});
  const int edge = geometry.appendEdge(p0, p1);

  const GeometryBounds static_bounds = geometry.edgeBounds(edge);
  EXPECT_FALSE(static_bounds.empty);
  EXPECT_DOUBLE_EQ(static_bounds.lower.x, 0.0);
  EXPECT_DOUBLE_EQ(static_bounds.upper.z, 3.0);

  const std::vector<glm::dvec3> directions{
      glm::dvec3{0.0, -1.0, 0.0},
      glm::dvec3{2.0, 0.0, -4.0},
  };
  const GeometryBounds swept =
      geometry.trajectoryEdgeBounds(edge, directions, 0.5);
  EXPECT_DOUBLE_EQ(swept.lower.y, -0.5);
  EXPECT_DOUBLE_EQ(swept.upper.x, 2.0);
  EXPECT_DOUBLE_EQ(swept.lower.z, 1.0);
}

TEST(GeometryTable, StoresInstanceGeometryWithTransform)
{
  GlobalGeometryManager geometry;
  GeometryMeshDesc mesh;
  mesh.vertices = {
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
  };
  mesh.edges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{1, 2},
  };
  mesh.triangles = {
      std::array<int, 3>{0, 1, 2},
  };

  glm::dmat4 transform{1.0};
  transform[3] = glm::dvec4{10.0, 0.0, 0.0, 1.0};
  const GeometryInstanceId instance =
      geometry.appendInstance(SubsystemId{3}, 4, mesh, transform);

  EXPECT_EQ(geometry.pointRange(SubsystemId{3}).count, 3);
  EXPECT_EQ(geometry.edgeRange(SubsystemId{3}).count, 2);
  EXPECT_EQ(geometry.triangleRange(SubsystemId{3}).count, 1);
  EXPECT_EQ(geometry.instances[static_cast<size_t>(instance)].localInstanceId,
            4);

  const GeometryPointId point =
      geometry.localToGlobalPoint(SubsystemId{3}, 1);
  EXPECT_GE(point, 0);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).x, 11.0);

  glm::dmat4 moved{1.0};
  moved[3] = glm::dvec4{-2.0, 5.0, 0.0, 1.0};
  geometry.setInstanceTransform(instance, moved);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).x, -1.0);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).y, 5.0);

  const std::array stencil{point};
  EXPECT_TRUE(geometry.classify(stencil).hasInstanceGeometry);
}

}  // namespace
}  // namespace ksk::runtime
