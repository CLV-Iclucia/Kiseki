#include <HairSim/rod.h>

#include <gtest/gtest.h>

#include <cmath>

namespace ksk::hairsim {
namespace {

Rod makeStraightRod()
{
  RodMaterial material;
  material.density = 1.0;
  material.radius = 0.5;
  material.shearModulus = 3.0;
  material.youngsModulus = 0.0;
  material.rootStiffness = 0.0;

  return Rod({
      RodBlock(0.0, 0.0, 0.0, 0.0),
      RodBlock(1.0, 0.0, 0.0, 0.0),
      RodBlock(2.0, 0.0, 0.0, 0.0),
  }, material);
}

TEST(TwistingEnergy, AccumulatesThetaGradientAndHessian)
{
  Rod rod = makeStraightRod();
  rod.state().setTheta(0, 0.2);
  rod.state().setTheta(1, 0.7);

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  const double twist = 0.5;
  const double twist_weight = rod.material().twistStiffness();
  EXPECT_DOUBLE_EQ(evaluation.energy.twisting,
                   0.5 * twist_weight * twist * twist);

  EXPECT_NEAR(evaluation.gradient[0].w, -twist_weight * twist, 1e-12);
  EXPECT_NEAR(evaluation.gradient[1].w, twist_weight * twist, 1e-12);
  EXPECT_NEAR(evaluation.gradient[2].w, 0.0, 1e-12);

  EXPECT_NEAR(evaluation.hessian.coeff(3, 3), twist_weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(3, 7), -twist_weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(7, 3), -twist_weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(7, 7), twist_weight, 1e-12);
}

TEST(TwistingEnergy, AccumulatesPositionJacobianFromCurvatureBinormal)
{
  Rod rod = makeStraightRod();
  rod.state().setPosition(1, glm::dvec3(1.0, 1.0, 0.0));
  rod.state().setTheta(1, 0.5);

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  const double twist = rod.materialTwist(1) - rod.restState().metrics[1].w;
  const double twist_weight =
      rod.material().twistStiffness() / rod.restState().metrics[1].z;
  const double position_jacobian = 1.0 / std::sqrt(2.0);

  EXPECT_NEAR(evaluation.gradient[0].z,
              twist_weight * twist * position_jacobian, 1e-12);
  EXPECT_NEAR(evaluation.gradient[1].z, 0.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[2].z,
              -twist_weight * twist * position_jacobian, 1e-12);

  EXPECT_NEAR(evaluation.hessian.coeff(2, 2),
              twist_weight * position_jacobian * position_jacobian, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(2, 3),
              -twist_weight * position_jacobian, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(10, 7),
              -twist_weight * position_jacobian, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(2, 10),
              -twist_weight * position_jacobian * position_jacobian, 1e-12);
}

TEST(TwistingEnergy, MaterialTwistIncludesTransportedReferenceTwist)
{
  Rod rod = makeStraightRod();
  const RodState previous_state = rod.state();

  rod.state().setPosition(1, glm::dvec3(1.0, 1.0, 0.0));
  rod.transportReferenceFrames(previous_state);

  EXPECT_NEAR(rod.materialTwist(1),
              rod.state().theta(1) - rod.state().theta(0) +
                  rod.referenceTwist(1),
              1e-12);

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  const double twist = rod.materialTwist(1) - rod.restState().metrics[1].w;
  const double twist_weight =
      rod.material().twistStiffness() / rod.restState().metrics[1].z;
  EXPECT_NEAR(evaluation.energy.twisting,
              0.5 * twist_weight * twist * twist, 1e-12);
}

TEST(TwistingEnergy, RestTwistHasZeroTwistingContribution)
{
  Rod rod = makeStraightRod();

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  EXPECT_DOUBLE_EQ(evaluation.energy.twisting, 0.0);
  EXPECT_DOUBLE_EQ(evaluation.gradient[0].w, 0.0);
  EXPECT_DOUBLE_EQ(evaluation.gradient[1].w, 0.0);
  EXPECT_DOUBLE_EQ(evaluation.gradient[2].w, 0.0);
}

}  // namespace
}  // namespace ksk::hairsim
