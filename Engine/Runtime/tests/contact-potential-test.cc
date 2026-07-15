#include <Runtime/contact-potential.h>

#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

TEST(ContactPotential, EvaluatesPointPointPenaltyEnergy)
{
  GlobalGeometryManager geometry;
  const GeometryPointId p0 =
      geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const GeometryPointId p1 =
      geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3(0.5, 0.0, 0.0));

  ContactTable contacts;
  contacts.stencils.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  EXPECT_DOUBLE_EQ(evaluateContactEnergy(geometry, contacts), 1.25);
}

TEST(ContactPotential, EvaluatesPointPointPenaltyGradient)
{
  GlobalGeometryManager geometry;
  const GeometryPointId p0 =
      geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const GeometryPointId p1 =
      geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3(0.5, 0.0, 0.0));

  ContactTable contacts;
  contacts.stencils.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  const ContactPotentialGradient gradient =
      evaluateContactGradient(geometry, contacts);

  ASSERT_EQ(gradient.points.size(), 2);
  EXPECT_EQ(gradient.points[0], p0);
  EXPECT_EQ(gradient.points[1], p1);
  EXPECT_DOUBLE_EQ(gradient.gradient.cpu()[0].x, 5.0);
  EXPECT_DOUBLE_EQ(gradient.gradient.cpu()[0].y, 0.0);
  EXPECT_DOUBLE_EQ(gradient.gradient.cpu()[1].x, -5.0);
  EXPECT_DOUBLE_EQ(gradient.gradient.cpu()[1].y, 0.0);
}

TEST(ContactPotential, IgnoresInactivePointPointStencil)
{
  GlobalGeometryManager geometry;
  const GeometryPointId p0 =
      geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const GeometryPointId p1 =
      geometry.appendPoint(SubsystemId{1}, 0, glm::dvec3(2.0, 0.0, 0.0));

  ContactTable contacts;
  contacts.stencils.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  const ContactPotentialGradient gradient =
      evaluateContactGradient(geometry, contacts);
  EXPECT_DOUBLE_EQ(evaluateContactEnergy(geometry, contacts), 0.0);
  EXPECT_TRUE(gradient.points.empty());
}

}  // namespace
}  // namespace ksk::runtime
