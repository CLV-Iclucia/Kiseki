#include <Runtime/contact-detection.h>

#include <gtest/gtest.h>

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
  geometry.addTriangle(p1, p2, p3);
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const GlobalContactRouter routed =
      runCCD(geometry, direction);

  ASSERT_EQ(routed.globalContacts.size(), 1);
  EXPECT_EQ(routed.globalContacts[0].type, ContactCase::PP);
  EXPECT_TRUE(routed.subsystemInternalContacts.empty());
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
  geometry.addTriangle(p1, p2, p3);
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const GlobalContactRouter routed =
      runCCD(geometry, direction);

  EXPECT_TRUE(routed.globalContacts.empty());
  ASSERT_EQ(routed.subsystemInternalContacts.size(), 1);
  EXPECT_EQ(routed.subsystemInternalContacts[0].subsystem, SubsystemId{3});
  ASSERT_EQ(routed.subsystemInternalContacts[0].contacts.size(), 1);
  EXPECT_EQ(routed.subsystemInternalContacts[0].contacts[0].type,
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
  geometry.addEdge(a0, a1);
  geometry.addEdge(b0, b1);

  auto direction = GeometryBuffer::CPU(geometry.pointCount());

  const GlobalContactRouter routed = runCCD(geometry, direction);

  ASSERT_EQ(routed.globalContacts.size(), 1);
  EXPECT_EQ(routed.globalContacts[0].type, ContactCase::EE);
  EXPECT_TRUE(routed.subsystemInternalContacts.empty());
}

}  // namespace
}  // namespace ksk::runtime
