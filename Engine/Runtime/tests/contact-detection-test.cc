#include <Runtime/contact-detection.h>

#include <gtest/gtest.h>

#include <algorithm>

namespace ksk::runtime {
namespace {

TEST(ContactDetection, RoutesCrossSubsystemPointPointToGlobal)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(2.0, 0.0, 0.0));
  const PointIdx p2 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(2.0, 1.0, 0.0));
  const PointIdx p3 =
      geometry.addPoint(SubsystemId{1}, 2, glm::dvec3(2.0, 0.0, 1.0));
  [[maybe_unused]] const int triangle = geometry.addTriangle(p1, p2, p3);
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const GlobalContactRouter routed =
      runCCD(geometry, direction);

  ASSERT_EQ(routed.globalContacts.size(), 1);
  EXPECT_EQ(routed.globalContacts[0].type, ContactCase::PP);
  EXPECT_TRUE(routed.subsystemInternalContacts.empty());
}

TEST(ContactDetection, GathersTypedPointTriangleWork)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(2.0, 0.0, 0.0));
  const PointIdx p2 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(2.0, 1.0, 0.0));
  const PointIdx p3 =
      geometry.addPoint(SubsystemId{1}, 2, glm::dvec3(2.0, 0.0, 1.0));
  const int triangle = geometry.addTriangle(p1, p2, p3);
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const ContactWorkList work =
      gatherCollisionWorkListAlongDirection(geometry, direction);

  ASSERT_EQ(work.deformablePointTriangles.size(), 1);
  EXPECT_EQ(work.deformablePointTriangles[0][0], p0);
  EXPECT_EQ(work.deformablePointTriangles[0][1], triangle);
  EXPECT_TRUE(work.deformableEdgeEdges.empty());
}

TEST(ContactDetection, RoutesSameSubsystemPointPointToInternal)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{3}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{3}, 1, glm::dvec3(2.0, 0.0, 0.0));
  const PointIdx p2 =
      geometry.addPoint(SubsystemId{3}, 2, glm::dvec3(2.0, 1.0, 0.0));
  const PointIdx p3 =
      geometry.addPoint(SubsystemId{3}, 3, glm::dvec3(2.0, 0.0, 1.0));
  [[maybe_unused]] const int triangle = geometry.addTriangle(p1, p2, p3);
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const GlobalContactRouter routed =
      runCCD(geometry, direction);

  EXPECT_TRUE(routed.globalContacts.empty());
  ASSERT_EQ(routed.subsystemInternalContacts.size(), 1);
  const SubsystemContactData& internal =
      routed.subsystemInternalContacts.at(SubsystemId{3});
  EXPECT_EQ(internal.subsystem, SubsystemId{3});
  ASSERT_EQ(internal.contacts.size(), 1);
  EXPECT_EQ(internal.contacts[0].type,
            ContactCase::PP);
}

TEST(ContactDetection, EdgeEdgePassKeepsEdgeEdgeContactsOnly)
{
  GlobalGeometryManager geometry;
  const PointIdx a0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(-0.5, 0.0, 0.0));
  const PointIdx a1 =
      geometry.addPoint(SubsystemId{0}, 1, glm::dvec3(0.5, 0.0, 0.0));
  const PointIdx b0 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.0, -0.5, 0.0005));
  const PointIdx b1 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(0.0, 0.5, 0.0005));
  [[maybe_unused]] const int first = geometry.addEdge(a0, a1);
  [[maybe_unused]] const int second = geometry.addEdge(b0, b1);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());

  const GlobalContactRouter routed = runCCD(geometry, direction);

  ASSERT_EQ(routed.globalContacts.size(), 1);
  EXPECT_EQ(routed.globalContacts[0].type, ContactCase::EE);
  EXPECT_TRUE(routed.subsystemInternalContacts.empty());
}

TEST(ContactDetection, GathersTypedEdgeEdgeWork)
{
  GlobalGeometryManager geometry;
  const PointIdx a0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(-0.5, 0.0, 0.0));
  const PointIdx a1 =
      geometry.addPoint(SubsystemId{0}, 1, glm::dvec3(0.5, 0.0, 0.0));
  const PointIdx b0 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.0, -0.5, 0.0005));
  const PointIdx b1 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(0.0, 0.5, 0.0005));
  const int first = geometry.addEdge(a0, a1);
  const int second = geometry.addEdge(b0, b1);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());

  const ContactWorkList work =
      gatherCollisionWorkListAlongDirection(geometry, direction);

  EXPECT_TRUE(work.deformablePointTriangles.empty());
  ASSERT_EQ(work.deformableEdgeEdges.size(), 1);
  EXPECT_EQ(work.deformableEdgeEdges[0][0], first);
  EXPECT_EQ(work.deformableEdgeEdges[0][1], second);
}

TEST(ContactDetection, GathersCoplanarPointTriangleWorkWithLBVH)
{
  GlobalGeometryManager geometry;
  const PointIdx point =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.25, 0.25, 0.0));
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(1.0, 0.0, 0.0));
  const PointIdx p2 =
      geometry.addPoint(SubsystemId{1}, 2, glm::dvec3(0.0, 1.0, 0.0));
  const int triangle = geometry.addTriangle(p0, p1, p2);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());

  const ContactWorkList work =
      gatherCollisionWorkListAlongDirection(geometry, direction);

  ASSERT_EQ(work.deformablePointTriangles.size(), 1);
  EXPECT_EQ(work.deformablePointTriangles[0][0], point);
  EXPECT_EQ(work.deformablePointTriangles[0][1], triangle);
}

TEST(ContactDetection, GathersSingleEdgeWorkWithLBVH)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{0}, 1, glm::dvec3(1.0, 0.0, 0.0));
  [[maybe_unused]] const int edge = geometry.addEdge(p0, p1);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());

  const ContactWorkList work =
      gatherCollisionWorkListAlongDirection(geometry, direction);

  EXPECT_TRUE(work.deformablePointTriangles.empty());
  EXPECT_TRUE(work.deformableEdgeEdges.empty());
}

TEST(ContactDetection, GathersPointTriangleWorkInStableOrder)
{
  GlobalGeometryManager geometry;
  const PointIdx query0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx query1 =
      geometry.addPoint(SubsystemId{0}, 1, glm::dvec3(1.0, 0.0, 0.0));
  const PointIdx a0 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.0, 0.0, 0.1));
  const PointIdx a1 =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(0.0, 1.0, 0.1));
  const PointIdx a2 =
      geometry.addPoint(SubsystemId{1}, 2, glm::dvec3(1.0, 0.0, 0.1));
  const PointIdx b0 =
      geometry.addPoint(SubsystemId{1}, 3, glm::dvec3(1.0, 0.0, 0.2));
  const PointIdx b1 =
      geometry.addPoint(SubsystemId{1}, 4, glm::dvec3(1.0, 1.0, 0.2));
  const PointIdx b2 =
      geometry.addPoint(SubsystemId{1}, 5, glm::dvec3(2.0, 0.0, 0.2));
  const int triangle0 = geometry.addTriangle(a0, a1, a2);
  const int triangle1 = geometry.addTriangle(b0, b1, b2);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  ContactDetectionConfig config;
  config.detectionDistance = 10.0;
  config.dHat = 10.0;

  const ContactWorkList work =
      gatherCollisionWorkListAlongDirection(geometry, direction, config);

  EXPECT_TRUE(std::ranges::is_sorted(work.deformablePointTriangles));
  EXPECT_EQ(std::ranges::adjacent_find(work.deformablePointTriangles),
            work.deformablePointTriangles.end());
  EXPECT_TRUE(std::ranges::binary_search(
      work.deformablePointTriangles, std::array<int, 2>{query0, triangle0}));
  EXPECT_TRUE(std::ranges::binary_search(
      work.deformablePointTriangles, std::array<int, 2>{query0, triangle1}));
  EXPECT_TRUE(std::ranges::binary_search(
      work.deformablePointTriangles, std::array<int, 2>{query1, triangle0}));
  EXPECT_TRUE(std::ranges::binary_search(
      work.deformablePointTriangles, std::array<int, 2>{query1, triangle1}));
}

}  // namespace
}  // namespace ksk::runtime
