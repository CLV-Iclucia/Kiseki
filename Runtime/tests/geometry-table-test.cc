#include <Runtime/geometry-table.h>

#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

TEST(GeometryTable, PreservesPointOwnership)
{
  GeometryTable geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{4}, 10, glm::dvec3{1.0, 0.0, 0.0});
  const auto p1 = geometry.appendPoint(SubsystemId{4}, 11, glm::dvec3{2.0, 0.0, 0.0});

  EXPECT_TRUE(geometry.contains(p0));
  EXPECT_TRUE(geometry.contains(p1));
  EXPECT_EQ(geometry.points[p0.value].subsystem, SubsystemId{4});
  EXPECT_EQ(geometry.points[p1.value].localSampleId, 11);
}

TEST(GeometryTable, AppendsEdgesAndTrianglesWithStableIds)
{
  GeometryTable geometry;

  const auto p0 = geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{0.0});
  const auto p1 = geometry.appendPoint(SubsystemId{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 = geometry.appendPoint(SubsystemId{0}, 2, glm::dvec3{0.0, 1.0, 0.0});

  const int edge_id = geometry.appendEdge(p0, p1, 0.05);
  const int triangle_id = geometry.appendTriangle(p0, p1, p2);

  EXPECT_EQ(edge_id, 0);
  EXPECT_EQ(triangle_id, 0);
  EXPECT_EQ(geometry.edges[edge_id].p0, p0);
  EXPECT_DOUBLE_EQ(geometry.edges[edge_id].radius, 0.05);
  EXPECT_EQ(geometry.triangles[triangle_id].p2, p2);
}

}  // namespace
}  // namespace ksk::runtime
