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

ClothMeshDesc makeSquareCloth()
{
  ClothMeshDesc mesh;
  mesh.vertices = {
      glm::dvec3(0.0, 0.0, 0.0),
      glm::dvec3(1.0, 0.0, 0.0),
      glm::dvec3(0.0, 1.0, 0.0),
      glm::dvec3(1.0, 1.0, 0.0),
  };
  mesh.triangles = {
      std::array<int, 3>{0, 1, 2},
      std::array<int, 3>{1, 3, 2},
  };
  mesh.material.arealDensity = 2.0;
  mesh.material.thickness = 0.05;
  mesh.material.stretchStiffness = 100.0;
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
  EXPECT_EQ(geometry.tets.size(), 1);
  EXPECT_EQ(geometry.points[2].localSampleId, 2);
}

TEST(FEMSubsystem, RemapsMultipleMeshTopologyIntoRuntimeGeometry)
{
  TetMeshDesc first = makeSingleTet();
  TetMeshDesc second = makeSingleTet();
  for (glm::dvec3& vertex : second.vertices) {
    vertex += glm::dvec3(10.0, 0.0, 0.0);
  }
  second.surfaceTriangles = {
      std::array<int, 3>{0, 2, 3},
  };
  second.tets = {
      std::array<int, 4>{3, 2, 1, 0},
  };

  FEMSubsystem subsystem(runtime::SubsystemId{3}, {first, second});
  runtime::GlobalGeometryManager geometry;

  subsystem.declareGeometry(geometry);

  ASSERT_EQ(geometry.pointCount(), 8);
  ASSERT_EQ(geometry.triangleCount(), 5);
  ASSERT_EQ(geometry.tetCount(), 2);
  ASSERT_EQ(subsystem.runtimeMeshes().size(), 2);

  const FEMMeshRuntimeRef& second_ref = subsystem.runtimeMeshes()[1];
  EXPECT_EQ(second_ref.points.first, 4);
  EXPECT_EQ(second_ref.points.count, 4);
  EXPECT_EQ(second_ref.transfer.localToGlobalPoint[0], 4);
  EXPECT_EQ(second_ref.transfer.localToGlobalPoint[3], 7);

  const auto triangle =
      geometry.globalTriangle(second_ref.transfer.localToGlobalTriangle[0]);
  EXPECT_EQ(triangle[0], 4);
  EXPECT_EQ(triangle[1], 6);
  EXPECT_EQ(triangle[2], 7);

  const auto tet = geometry.globalTet(second_ref.transfer.localToGlobalTet[0]);
  EXPECT_EQ(tet[0], 7);
  EXPECT_EQ(tet[1], 6);
  EXPECT_EQ(tet[2], 5);
  EXPECT_EQ(tet[3], 4);
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
  subsystem.mapLocalDirectionToGeometry(dq.view(), dx.view());

  EXPECT_DOUBLE_EQ(dx.cpu()[1].x, 1.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].y, 2.0);
  EXPECT_DOUBLE_EQ(dx.cpu()[1].z, 3.0);
}

TEST(FEMSubsystem, EmitsClothSurfaceGeometry)
{
  runtime::RuntimeSceneDesc scene;
  const runtime::ObjectRef cloth_ref = addClothMesh(scene, makeSquareCloth());

  runtime::SimulationContext simulation = runtime::buildSimulation(scene);

  EXPECT_EQ(simulation.scene().dofs.totalScalars, 12);
  EXPECT_EQ(simulation.scene().geometry.points.size(), 4);
  EXPECT_EQ(simulation.scene().geometry.edges.size(), 5);
  EXPECT_EQ(simulation.scene().geometry.triangles.size(), 2);
  EXPECT_EQ(simulation.scene().geometry.tets.size(), 0);
  EXPECT_DOUBLE_EQ(simulation.scene().geometry.points[0].radius, 0.05);
  EXPECT_DOUBLE_EQ(simulation.scene().geometry.edges[0].radius, 0.05);
  EXPECT_DOUBLE_EQ(simulation.scene().geometry.triangles[0].thickness, 0.05);

  const std::vector<runtime::PropertyDescriptor> properties =
      scene.listProperties(cloth_ref);
  EXPECT_TRUE(std::any_of(properties.begin(), properties.end(),
                         [&](const runtime::PropertyDescriptor& property) {
                           return property.name == "material.arealDensity" &&
                                  property.ownerId == cloth_ref.id;
                         }));
  EXPECT_TRUE(std::any_of(properties.begin(), properties.end(),
                         [&](const runtime::PropertyDescriptor& property) {
                           return property.name == "material.thickness" &&
                                  property.ownerId == cloth_ref.id;
                         }));
}

TEST(FEMSubsystem, MixedTetAndClothHaveContiguousOffsets)
{
  runtime::RuntimeSceneDesc scene;
  addTetMesh(scene, makeSingleTet());
  addClothMesh(scene, makeSquareCloth());

  runtime::SimulationContext simulation = runtime::buildSimulation(scene);

  EXPECT_EQ(simulation.scene().dofs.totalScalars, 24);
  EXPECT_EQ(simulation.scene().geometry.points.size(), 8);
  EXPECT_EQ(simulation.scene().geometry.triangles.size(), 6);
  EXPECT_EQ(simulation.scene().geometry.tets.size(), 1);
}

TEST(FEMSubsystem, DeformedClothHasElasticGradient)
{
  ClothMeshDesc mesh = makeSquareCloth();
  mesh.initialPositions = mesh.vertices;
  mesh.initialPositions[1] = glm::dvec3(1.2, 0.0, 0.0);
  FEMSubsystem subsystem(
      runtime::SubsystemId{0},
      std::vector<FEMPrimitive>{
          FEMClothPrimitive{
              .mesh = mesh,
              .offset = {},
              .runtime = {},
          },
      },
      {},
      0,
      glm::dvec3(0.0));
  auto q = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  auto qdot = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.writeState(q.view(), qdot.view());
  subsystem.beginStep(q.view(), qdot.view(), 0.01);
  subsystem.prepareLocalOperator(0.01);

  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g.view());

  EXPECT_GT(g.norm(), 1.0e-6);
}

TEST(FEMSubsystem, RestTetHasZeroElasticGradient)
{
  FEMSubsystem subsystem(runtime::SubsystemId{0}, {makeSingleTet()});
  auto q = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  auto qdot = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.writeState(q.view(), qdot.view());
  subsystem.beginStep(q.view(), qdot.view(), 0.01);
  subsystem.prepareLocalOperator(0.01);

  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g.view());

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
  subsystem.writeState(q.view(), qdot.view());
  subsystem.beginStep(q.view(), qdot.view(), 0.01);
  subsystem.prepareLocalOperator(0.01);

  auto g = runtime::DofBuffer::CPU(subsystem.localScalarCount());
  subsystem.assembleLocalGradient(g.view());

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

  runtime::SimulationContext simulation = runtime::buildSimulation(scene);
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
