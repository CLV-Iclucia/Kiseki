#include <Runtime/contact-barrier-energy.h>

#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

TEST(ContactPotential, EvaluatesPointPointGIPCEnergy)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.5, 0.0, 0.0));

  ContactStencils contacts;
  contacts.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  EXPECT_GT(computeContactEnergy(geometry, contacts), 0.0);
}

TEST(ContactPotential, EvaluatesPointPointGIPCGradient)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.5, 0.0, 0.0));

  ContactStencils contacts;
  contacts.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  const ContactPotentialGradient gradient =
      computeContactGradient(geometry, contacts);

  ASSERT_EQ(gradient.points.size(), 2);
  EXPECT_EQ(gradient.points[0], p0);
  EXPECT_EQ(gradient.points[1], p1);
  EXPECT_GT(gradient.gradient.cpu()[0].x, 0.0);
  EXPECT_LT(gradient.gradient.cpu()[1].x, 0.0);
  EXPECT_NEAR(gradient.gradient.cpu()[0].x + gradient.gradient.cpu()[1].x,
              0.0,
              1.0e-12);
  EXPECT_NEAR(gradient.gradient.cpu()[0].y, 0.0, 1.0e-12);
  EXPECT_NEAR(gradient.gradient.cpu()[1].y, 0.0, 1.0e-12);
}

TEST(ContactPotential, IgnoresInactivePointPointStencil)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(2.0, 0.0, 0.0));

  ContactStencils contacts;
  contacts.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  const ContactPotentialGradient gradient =
      computeContactGradient(geometry, contacts);
  EXPECT_DOUBLE_EQ(computeContactEnergy(geometry, contacts), 0.0);
  EXPECT_TRUE(gradient.points.empty());
}

TEST(ContactPotential, EvaluatesPointEdgeAndPointTriangleGIPCGradient)
{
  GlobalGeometryManager geometry;
  const PointIdx p =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.25, 0.25, 0.25));
  const PointIdx a =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx b =
      geometry.addPoint(SubsystemId{1}, 1, glm::dvec3(1.0, 0.0, 0.0));
  const PointIdx c =
      geometry.addPoint(SubsystemId{1}, 2, glm::dvec3(0.0, 1.0, 0.0));

  ContactStencils contacts;
  contacts.push_back(ContactStencil{
      .type = ContactCase::PE,
      .geometryIds = {p, a, b, -1},
      .dHat = 0.5,
      .stiffness = 2.0,
  });
  contacts.push_back(ContactStencil{
      .type = ContactCase::PT,
      .geometryIds = {p, a, b, c},
      .dHat = 0.5,
      .stiffness = 2.0,
  });

  const ContactPotentialGradient gradient =
      computeContactGradient(geometry, contacts);
  EXPECT_GT(computeContactEnergy(geometry, contacts), 0.0);
  EXPECT_FALSE(gradient.points.empty());
}

TEST(ContactPotential, EvaluatesAnalyticHessianProduct)
{
  GlobalGeometryManager geometry;
  const PointIdx p0 =
      geometry.addPoint(SubsystemId{0}, 0, glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx p1 =
      geometry.addPoint(SubsystemId{1}, 0, glm::dvec3(0.5, 0.0, 0.0));

  ContactStencils contacts;
  contacts.push_back(ContactStencil{
      .type = ContactCase::PP,
      .geometryIds = {p0, p1, -1, -1},
      .dHat = 1.0,
      .stiffness = 10.0,
  });

  GeometryBuffer direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(p0)] = glm::dvec3(0.1, 0.0, 0.0);
  direction.cpu()[static_cast<size_t>(p1)] = glm::dvec3(0.0, 0.0, 0.0);

  const ContactPotentialGradient product =
      computeContactHessianProduct(geometry, contacts, direction.view().asConst());
  ASSERT_EQ(product.points.size(), 2);
  EXPECT_NEAR(product.gradient.cpu()[0].x + product.gradient.cpu()[1].x,
              0.0,
              1.0e-10);
}

TEST(ContactPotential, EdgeEdgeMollifierUsesRestScale)
{
  GlobalGeometryManager small_rest_geometry;
  const PointIdx s0 = small_rest_geometry.addPoint(
      SubsystemId{0},
      0,
      glm::dvec3(0.0, 0.0, 0.0),
      glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx s1 = small_rest_geometry.addPoint(
      SubsystemId{0},
      1,
      glm::dvec3(1.0, 0.0, 0.0),
      glm::dvec3(0.001, 0.0, 0.0));
  const PointIdx s2 = small_rest_geometry.addPoint(
      SubsystemId{1},
      0,
      glm::dvec3(0.0, 0.1, 0.05),
      glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx s3 = small_rest_geometry.addPoint(
      SubsystemId{1},
      1,
      glm::dvec3(1.0, 0.1001, 0.05),
      glm::dvec3(0.0, 0.001, 0.0));

  GlobalGeometryManager large_rest_geometry;
  const PointIdx l0 = large_rest_geometry.addPoint(
      SubsystemId{0},
      0,
      glm::dvec3(0.0, 0.0, 0.0),
      glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx l1 = large_rest_geometry.addPoint(
      SubsystemId{0},
      1,
      glm::dvec3(1.0, 0.0, 0.0),
      glm::dvec3(1.0, 0.0, 0.0));
  const PointIdx l2 = large_rest_geometry.addPoint(
      SubsystemId{1},
      0,
      glm::dvec3(0.0, 0.1, 0.05),
      glm::dvec3(0.0, 0.0, 0.0));
  const PointIdx l3 = large_rest_geometry.addPoint(
      SubsystemId{1},
      1,
      glm::dvec3(1.0, 0.1001, 0.05),
      glm::dvec3(0.0, 1.0, 0.0));

  ContactStencils small_rest_contacts;
  small_rest_contacts.push_back(ContactStencil{
      .type = ContactCase::EE,
      .geometryIds = {s0, s1, s2, s3},
      .dHat = 0.5,
      .stiffness = 2.0,
  });
  ContactStencils large_rest_contacts;
  large_rest_contacts.push_back(ContactStencil{
      .type = ContactCase::EE,
      .geometryIds = {l0, l1, l2, l3},
      .dHat = 0.5,
      .stiffness = 2.0,
  });

  const double small_rest_energy =
      computeContactEnergy(small_rest_geometry, small_rest_contacts);
  const double large_rest_energy =
      computeContactEnergy(large_rest_geometry, large_rest_contacts);

  EXPECT_GT(small_rest_energy, 0.0);
  EXPECT_GT(large_rest_energy, 0.0);
  EXPECT_LT(large_rest_energy, small_rest_energy * 1.0e-3);
}

}  // namespace
}  // namespace ksk::runtime
