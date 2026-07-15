#include <Runtime/contact-detection.h>

#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

TEST(ContactDetection, RoutesCrossSubsystemPointPointToGlobal)
{
  GlobalGeometryManager geometry;
  const GeometryPointId p0 =
      geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const GeometryPointId p1 =
      geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3(2.0, 0.0, 0.0));
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const RoutedContactTables routed =
      detectContactsAlongDirection(geometry, direction);

  ASSERT_EQ(routed.globalContacts.stencils.size(), 1);
  EXPECT_EQ(routed.globalContacts.stencils[0].type, ContactCase::PP);
  EXPECT_TRUE(routed.internalContacts.empty());
  (void)p1;
}

TEST(ContactDetection, RoutesSameSubsystemPointPointToInternal)
{
  GlobalGeometryManager geometry;
  const GeometryPointId p0 =
      geometry.appendPoint(SubsystemId{3}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const GeometryPointId p1 =
      geometry.appendPoint(SubsystemId{3}, 1, glm::dvec3(2.0, 0.0, 0.0));
  auto direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(2.0, 0.0, 0.0);

  const RoutedContactTables routed =
      detectContactsAlongDirection(geometry, direction);

  EXPECT_TRUE(routed.globalContacts.stencils.empty());
  ASSERT_EQ(routed.internalContacts.size(), 1);
  EXPECT_EQ(routed.internalContacts[0].subsystem, SubsystemId{3});
  ASSERT_EQ(routed.internalContacts[0].contacts.stencils.size(), 1);
  EXPECT_EQ(routed.internalContacts[0].contacts.stencils[0].type,
            ContactCase::PP);
  (void)p1;
}

}  // namespace
}  // namespace ksk::runtime
