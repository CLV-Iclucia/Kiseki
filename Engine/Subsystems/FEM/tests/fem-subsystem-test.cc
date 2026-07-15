#include <FEM/fem-scene.h>
#include <FEM/fem-subsystem.h>

#include <Runtime/global-solver.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

namespace ksk::engine::fem {
namespace {

TetMeshDesc makeSingleTet()
{
  TetMeshDesc mesh;
  mesh.vertices = {
      glm::dvec3(0.0, 0.0, 0.0),
      glm::dvec3(1.0, 0.0, 0.0),
      glm::dvec3(0.0, 1.0, 0.0),
      glm::dvec3(0.0, 0.0, 1.0),
  };
  mesh.tets = {std::array<int, 4>{0, 1, 2, 3}};
  mesh.surfaceEdges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{0, 2},
      std::array<int, 2>{0, 3},
      std::array<int, 2>{1, 2},
      std::array<int, 2>{1, 3},
      std::array<int, 2>{2, 3},
  };
  mesh.surfaceTriangles = {
      std::array<int, 3>{0, 1, 2},
      std::array<int, 3>{0, 1, 3},
      std::array<int, 3>{0, 2, 3},
      std::array<int, 3>{1, 2, 3},
  };
  mesh.material.density = 2.0;
  return mesh;
}

TEST(FEMSubsystem, EmitsSurfaceGeometry)
{
  FEMSubsystem subsystem(runtime::SubsystemId{3}, {makeSingleTet()});
  runtime::GlobalGeometryManager geometry;

  subsystem.declareGeometry(geometry);

  EXPECT_EQ(subsystem.dofRange().scalarCount, 12);
  EXPECT_EQ(subsystem.dofRange().blockSize, 3);
  EXPECT_EQ(geometry.points.size(), 4);
  EXPECT_EQ(geometry.edges.size(), 6);
  EXPECT_EQ(geometry.triangles.size(), 4);
  EXPECT_EQ(geometry.points[0].subsystem, runtime::SubsystemId{3});
  EXPECT_EQ(geometry.points[2].localSampleId, 2);
}

TEST(FEMSubsystem, MapsPositionDirectionToGeometry)
{
  FEMSubsystem subsystem(runtime::SubsystemId{0}, {makeSingleTet()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  auto dq = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  dq.cpu()[3] = 1.0;
  dq.cpu()[4] = 2.0;
  dq.cpu()[5] = 3.0;

  auto dx = runtime::GeometryBuffer::CPU(4);
  subsystem.mapDirectionToGeometry(dq, dx);

  EXPECT_DOUBLE_EQ(dx.cpu()[1].x, 1.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].y, 2.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].z, 3.0);
}

TEST(FEMSubsystem, ContactGradientScattersToPositions)
{
  FEMSubsystem subsystem(runtime::SubsystemId{0}, {makeSingleTet()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  const std::array<runtime::GeometryPointId, 1> points{geometry.points[2].id};
  auto gradients =
      runtime::GeometryBuffer::FromCPU({glm::dvec3(4.0, 5.0, 6.0)});
  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());

  subsystem.scatterContactGradient(points, gradients, g);

  EXPECT_DOUBLE_EQ(g.cpu()[6], 4.0);
  EXPECT_DOUBLE_EQ(g.cpu()[7], 5.0);
  EXPECT_DOUBLE_EQ(g.cpu()[8], 6.0);
}

TEST(FEMSubsystem, RestTetHasZeroElasticGradient)
{
  FEMSubsystem subsystem(runtime::SubsystemId{0}, {makeSingleTet()});
  auto q = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  auto qdot = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.writeState(q, qdot);
  subsystem.beginStep(q, qdot, 0.01);
  subsystem.prepareLocalOperator(0.01);

  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g);

  EXPECT_NEAR(g.norm(), 0.0, 1.0e-8);
}

TEST(FEMSubsystem, DeformedTetHasElasticGradient)
{
  TetMeshDesc mesh = makeSingleTet();
  mesh.initialPositions = mesh.vertices;
  mesh.initialPositions[1] = glm::dvec3(1.2, 0.0, 0.0);
  FEMSubsystem subsystem(runtime::SubsystemId{0}, {mesh});
  auto q = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  auto qdot = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.writeState(q, qdot);
  subsystem.beginStep(q, qdot, 0.01);
  subsystem.prepareLocalOperator(0.01);

  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g);

  EXPECT_GT(g.norm(), 1.0e-6);
}

TEST(FEMSceneFrontend, BuildsRuntimeAndListsProperties)
{
  runtime::RuntimeSceneDesc scene;
  const runtime::ObjectRef mesh_ref = addTetMesh(scene, makeSingleTet());
  addConstraint(scene, mesh_ref, "y", 0, 100.0,
                runtime::ScalarConstraintTarget(
                    [](double) { return 0.25; }));

  const std::vector<runtime::PropertyDescriptor> properties =
      scene.listProperties(mesh_ref);
  const auto has_property = [&](std::string_view name) {
    return std::any_of(properties.begin(), properties.end(),
                       [&](const runtime::PropertyDescriptor& property) {
                         return property.name == name &&
                                property.ownerId == mesh_ref.id;
                       });
  };

  EXPECT_TRUE(has_property("material.density"));
  EXPECT_TRUE(has_property("x"));
  EXPECT_TRUE(has_property("y"));
  EXPECT_TRUE(has_property("z"));
  EXPECT_EQ(scene.constraints.size(), 1);

  runtime::RuntimeSimulation simulation = runtime::buildSimulation(scene);
  EXPECT_EQ(simulation.scene().dofs.totalScalars, 12);
  EXPECT_EQ(simulation.scene().geometry.points.size(), 4);
  EXPECT_EQ(simulation.scene().geometry.triangles.size(), 4);
}

TEST(FEMSceneFrontend, RunsPenaltyConstraintStep)
{
  runtime::RuntimeSceneDesc scene;
  scene.gravity = glm::dvec3(0.0);
  scene.timeStep = 0.01;
  scene.solverConfig.maxNewtonIterations = 5;
  runtime::ObjectRef mesh_ref = addTetMesh(scene, makeSingleTet());
  addConstraint(scene, mesh_ref, "x", 0, 100.0,
                runtime::ScalarConstraintTarget(
                    [](double) { return 0.0; }));

  runtime::SimulationRunner runner = runtime::buildSimulationRunner(scene);
  const runtime::RuntimeStepResult result = runner.step();

  EXPECT_TRUE(result.converged);
  EXPECT_EQ(runner.simulation().scene().dofs.totalScalars, 12);
}

}  // namespace
}  // namespace ksk::engine::fem
