#include <DER/der-subsystem.h>
#include <DER/der-scene.h>
#include <Runtime/global-solver.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace ksk::der {
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

TEST(DERRod, TwistingEnergyMatchesHairSimReferenceCase)
{
  Rod rod = makeStraightRod();
  rod.state().setTheta(0, 0.2);
  rod.state().setTheta(1, 0.7);

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  const double twist = 0.5;
  const double weight = rod.material().twistStiffness();
  EXPECT_DOUBLE_EQ(evaluation.energy.twisting, 0.5 * weight * twist * twist);
  EXPECT_NEAR(evaluation.gradient[0].w, -weight * twist, 1e-12);
  EXPECT_NEAR(evaluation.gradient[1].w, weight * twist, 1e-12);
  EXPECT_NEAR(evaluation.gradient[2].w, 0.0, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(3, 3), weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(3, 7), -weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(7, 3), -weight, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(7, 7), weight, 1e-12);
}

TEST(DERRod, ConstrainsPositionAndTwistProperties)
{
  RodMaterial material;
  material.density = 1.0;
  material.youngsModulus = 0.0;
  material.shearModulus = 0.0;

  Rod rod({
      RodBlock(0.0, 0.0, 0.0, 0.0),
      RodBlock(1.0, 0.0, 0.0, 0.0),
      RodBlock(2.0, 0.0, 0.0, 0.0),
  }, material);
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::X,
      .sample = 0,
      .target = 0.0,
      .stiffness = 10.0,
  });
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::Y,
      .sample = 0,
      .target = 0.0,
      .stiffness = 10.0,
  });
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::Z,
      .sample = 0,
      .target = 0.0,
      .stiffness = 10.0,
  });
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::Y,
      .sample = 2,
      .target = 0.0,
      .stiffness = 10.0,
  });
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::Twist,
      .sample = 0,
      .target = 0.0,
      .stiffness = 10.0,
  });
  rod.addConstraint(RodConstraint{
      .property = RodConstraintProperty::Twist,
      .sample = 1,
      .target = 0.25,
      .stiffness = 10.0,
  });
  rod.state().setPosition(0, glm::dvec3(0.1, 0.2, 0.3));
  rod.state().setTheta(0, 0.4);
  rod.state().setPosition(2, glm::dvec3(2.0, 0.5, 0.0));
  rod.state().setTheta(1, 0.75);

  const RodEvaluation evaluation = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(evaluation.valid) << evaluation.diagnostic;

  EXPECT_NEAR(evaluation.energy.constraint, 4.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[0].x, 1.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[0].y, 2.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[0].z, 3.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[0].w, 4.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[1].w, 5.0, 1e-12);
  EXPECT_NEAR(evaluation.gradient[2].y, 5.0, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(0, 0), 10.0, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(3, 3), 10.0, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(7, 7), 10.0, 1e-12);
  EXPECT_NEAR(evaluation.hessian.coeff(9, 9), 10.0, 1e-12);
}

TEST(DERSubsystem, EmitsCenterlineGeometry)
{
  DERSubsystem subsystem(runtime::SubsystemId{2}, {makeStraightRod()});
  runtime::GlobalGeometryManager geometry;

  subsystem.declareGeometry(geometry);

  EXPECT_EQ(subsystem.dofRange().scalarCount, 12);
  EXPECT_EQ(geometry.points.size(), 3);
  EXPECT_EQ(geometry.edges.size(), 2);
  EXPECT_TRUE(geometry.triangles.empty());
  EXPECT_EQ(geometry.points[0].subsystem, runtime::SubsystemId{2});
  EXPECT_EQ(geometry.points[1].localSampleId, 1);
  EXPECT_DOUBLE_EQ(geometry.edges[0].radius, 0.5);
}

TEST(DERSubsystem, MapsOnlyPositionDirectionToGeometry)
{
  DERSubsystem subsystem(runtime::SubsystemId{0}, {makeStraightRod()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  auto dq = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  dq.cpu()[0] = 1.0;
  dq.cpu()[1] = 2.0;
  dq.cpu()[2] = 3.0;
  dq.cpu()[3] = 10.0;
  dq.cpu()[7] = 20.0;

  auto dx = runtime::GeometryBuffer::CPU(3);
  subsystem.mapDirectionToGeometry(dq, dx);

  EXPECT_DOUBLE_EQ(dx.cpu()[0].x, 1.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[0].y, 2.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[0].z, 3.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].x, 0.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].y, 0.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].z, 0.0);
}

TEST(DERSubsystem, ContactGradientScattersOnlyToPositions)
{
  DERSubsystem subsystem(runtime::SubsystemId{0}, {makeStraightRod()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  const std::array<runtime::GeometryPointId, 1> points{geometry.points[1].id};
  auto gradients =
      runtime::GeometryBuffer::FromCPU({glm::dvec3(4.0, 5.0, 6.0)});
  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());

  subsystem.scatterContactGradient(points, gradients, g);

  EXPECT_DOUBLE_EQ(g.cpu()[4], 4.0);
  EXPECT_DOUBLE_EQ(g.cpu()[5], 5.0);
  EXPECT_DOUBLE_EQ(g.cpu()[6], 6.0);
  EXPECT_DOUBLE_EQ(g.cpu()[7], 0.0);
}

TEST(DERSubsystem, UsesRodEvaluationForLocalGradient)
{
  Rod rod = makeStraightRod();
  rod.state().setTheta(0, 0.2);
  rod.state().setTheta(1, 0.7);
  const RodEvaluation direct = rod.evaluate(glm::dvec3(0.0));
  ASSERT_TRUE(direct.valid) << direct.diagnostic;

  DERSubsystem subsystem(runtime::SubsystemId{0}, {rod});
  subsystem.prepareLocalOperator(0.01);
  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g);

  for (int vertex = 0; vertex < 3; ++vertex) {
    for (int lane = 0; lane < 4; ++lane) {
      EXPECT_NEAR(g.cpu()[4 * vertex + lane],
                  direct.gradient[vertex][lane],
                  1e-12);
    }
  }
}

class CountingVisitor final : public runtime::SubsystemBackendVisitor {
 public:
  void visit(DERSubsystem& subsystem) override
  {
    visited = subsystem.id();
  }

  int visited = -1;
};

TEST(DERSubsystem, AcceptsSubsystemBackendVisitor)
{
  DERSubsystem subsystem(runtime::SubsystemId{9}, {makeStraightRod()});
  CountingVisitor visitor;

  subsystem.accept(visitor);

  EXPECT_EQ(visitor.visited, 9);
}

TEST(DERSceneFrontend, BuildsRuntimeAndRunsNoContactNewtonStep)
{
  DERRodDesc rod;
  rod.restBlocks = {
      RodBlock(0.0, 0.0, 0.0, 0.0),
      RodBlock(1.0, 0.0, 0.0, 0.0),
      RodBlock(2.0, 0.0, 0.0, 0.0),
  };
  rod.material.rootStiffness = 0.0;

  runtime::RuntimeSceneDesc scene;
  scene.gravity = glm::dvec3(0.0);
  scene.timeStep = 0.01;
  scene.solverConfig.maxNewtonIterations = 3;
  addRod(scene, rod);

  runtime::SimulationRunner simulation =
      runtime::buildSimulationRunner(scene);
  const runtime::RuntimeStepResult result = simulation.step();

  EXPECT_EQ(simulation.simulation().scene().dofs.totalScalars, 12);
  EXPECT_EQ(simulation.simulation().scene().geometry.points.size(), 3);
  EXPECT_TRUE(result.converged);
}

TEST(DERSceneFrontend, RodObjectListsConfigurableProperties)
{
  DERRodDesc rod;
  rod.restBlocks = {
      RodBlock(0.0, 0.0, 0.0, 0.0),
      RodBlock(1.0, 0.0, 0.0, 0.0),
      RodBlock(2.0, 0.0, 0.0, 0.0),
  };

  runtime::RuntimeSceneDesc scene;
  const runtime::ObjectRef rodRef = addRod(scene, rod);
  addConstraint(scene, rodRef, "x", 0, 100.0,
                runtime::ScalarConstraintTarget(
                    [](double time) { return time; }));
  addConstraint(scene, rodRef, "twist", 0, 50.0,
                runtime::ScalarConstraintTarget(
                    [](double) { return 0.0; }));
  const std::vector<runtime::PropertyDescriptor> properties =
      scene.listProperties(rodRef);

  const auto hasProperty = [&](std::string_view name) {
    return std::any_of(properties.begin(), properties.end(),
                       [&](const runtime::PropertyDescriptor& property) {
                         return property.name == name &&
                                property.ownerId == rodRef.id;
                       });
  };

  EXPECT_TRUE(hasProperty("material.youngsModulus"));
  EXPECT_TRUE(hasProperty("material.shearModulus"));
  EXPECT_TRUE(hasProperty("x"));
  EXPECT_TRUE(hasProperty("y"));
  EXPECT_TRUE(hasProperty("z"));
  EXPECT_TRUE(hasProperty("twist"));
  EXPECT_EQ(scene.constraints.size(), 2);
  EXPECT_EQ(scene.constraints[0].object.id, rodRef.id);
  EXPECT_EQ(scene.constraints[0].property, "x");
  EXPECT_EQ(scene.constraints[1].property, "twist");
}

}  // namespace
}  // namespace ksk::der
