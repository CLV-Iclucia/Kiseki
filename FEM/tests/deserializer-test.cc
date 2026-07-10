//
// Created by creeper on 10/31/24.
//

#include <gtest/gtest.h>
#include <Core/deserializer.h>
#include <Core/json.h>
#include <fem/primitives/tet-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitive.h>
#include <fem/system.h>
#include <fem/integrator.h>
#include <fem/fem-simulation.h>
#include <Deform/strain-energy-density.h>
#include <Maths/linear-solver.h>
#include <memory>

using ksk::core::JsonNode;
using ksk::core::JsonDict;
using ksk::core::JsonList;
using namespace ksk::fem;
using namespace ksk::deform;
using namespace ksk::maths;

TEST(FEMDeserializerTest, TetMeshDeserialization) {
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    JsonNode meshNode = JsonNode(meshDict);
    
    auto mesh = deserialize<TetMesh>(meshNode);
    
    EXPECT_EQ(mesh.getVertices().size(), 4);
    EXPECT_EQ(mesh.tets.size(), 1);
    
    // 验证顶点数据
    EXPECT_DOUBLE_EQ(mesh.getVertices()[0](0), 0.0);
    EXPECT_DOUBLE_EQ(mesh.getVertices()[0](1), 0.0);
    EXPECT_DOUBLE_EQ(mesh.getVertices()[0](2), 0.0);
    
    EXPECT_DOUBLE_EQ(mesh.getVertices()[1](0), 1.0);
    EXPECT_DOUBLE_EQ(mesh.getVertices()[1](1), 0.0);
    EXPECT_DOUBLE_EQ(mesh.getVertices()[1](2), 0.0);
    
    // 验证四面体数据
    EXPECT_EQ(mesh.tets[0][0], 0);
    EXPECT_EQ(mesh.tets[0][1], 1);
    EXPECT_EQ(mesh.tets[0][2], 2);
    EXPECT_EQ(mesh.tets[0][3], 3);
}

// 测试ElasticTetMesh反序列化
TEST(FEMDeserializerTest, ElasticTetMeshDeserialization) {
    // 创建TetMesh
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    // 创建StrainEnergyDensity
    JsonDict energyDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(8e5)},
        {"lambda", JsonNode(1.2e6)}
    };
    
    JsonDict elasticTetMeshDict{
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    JsonNode elasticTetMeshNode = JsonNode(elasticTetMeshDict);
    
    auto elasticTetMesh = deserialize<ElasticTetMesh>(elasticTetMeshNode);
    
    EXPECT_EQ(elasticTetMesh.mesh.getVertices().size(), 4);
    EXPECT_EQ(elasticTetMesh.mesh.tets.size(), 1);
    EXPECT_DOUBLE_EQ(elasticTetMesh.density, 1000.0);
    EXPECT_NE(elasticTetMesh.energy, nullptr);
}

// 测试Primitive反序列化
TEST(FEMDeserializerTest, PrimitiveDeserialization) {
    // 创建TetMesh
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    // 创建StrainEnergyDensity
    JsonDict energyDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(8e5)},
        {"lambda", JsonNode(1.2e6)}
    };
    
    JsonDict primitiveDict{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    JsonNode primitiveNode = JsonNode(primitiveDict);
    
    auto primitive = deserialize<Primitive>(primitiveNode);
    
    EXPECT_EQ(primitive.getVertexCount(), 4);
    EXPECT_EQ(primitive.dofDim(), 12); // 4个顶点 * 3个自由度
}

// 测试SystemConfig反序列化
TEST(FEMDeserializerTest, SystemConfigDeserialization) {
    // 创建primitives列表
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict energyDict{
        {"type", JsonNode(std::string("ARAP"))}
    };
    
    JsonDict primitiveDict{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    
    JsonDict systemConfigDict{
        {"primitives", JsonNode(JsonList{JsonNode(primitiveDict)})},
        {"colliders", JsonNode(JsonList{})}
    };
    JsonNode systemConfigNode = JsonNode(systemConfigDict);
    
    auto systemConfig = deserialize<SystemConfig>(systemConfigNode);
    
    EXPECT_EQ(systemConfig.primitives.size(), 1);
    EXPECT_EQ(systemConfig.colliders.size(), 0);
    EXPECT_EQ(systemConfig.primitives[0].getVertexCount(), 4);
}

// 测试System反序列化
TEST(FEMDeserializerTest, SystemDeserialization) {
    // 创建复杂的系统配置
    JsonDict meshDict1{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict meshDict2{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(2.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(3.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(2.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(2.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict energyDict1{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(8e5)},
        {"lambda", JsonNode(1.2e6)}
    };
    
    JsonDict energyDict2{
        {"type", JsonNode(std::string("ARAP"))}
    };
    
    JsonDict primitiveDict1{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict1)},
        {"energy", JsonNode(energyDict1)},
        {"density", JsonNode(1000.0)}
    };
    
    JsonDict primitiveDict2{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict2)},
        {"energy", JsonNode(energyDict2)},
        {"density", JsonNode(1200.0)}
    };
    
    JsonDict systemDict{
        {"primitives", JsonNode(JsonList{
            JsonNode(primitiveDict1),
            JsonNode(primitiveDict2)
        })},
        {"colliders", JsonNode(JsonList{})}
    };
    JsonNode systemNode = JsonNode(systemDict);
    
    SystemBuilder builder;
    auto system = builder.build(systemNode);
    
    EXPECT_EQ(system.primitives().size(), 2);
    EXPECT_EQ(system.dof(), 24); // 2个primitive，每个4个顶点，每个顶点3个自由度
    EXPECT_EQ(system.primitive(0).getVertexCount(), 4);
    EXPECT_EQ(system.primitive(1).getVertexCount(), 4);
}

// 测试多种能量密度函数的工厂函数
TEST(FEMDeserializerTest, MultipleEnergyDensityDeserialization) {
    // 测试StableNeoHookean
    JsonDict stableNeoHookeanDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(5e5)},
        {"lambda", JsonNode(8e5)}
    };
    JsonNode stableNeoHookeanNode = JsonNode(stableNeoHookeanDict);
    
    auto stableNeoHookean = createStrainEnergyDensity<Real>(stableNeoHookeanNode);
    EXPECT_NE(stableNeoHookean, nullptr);
    
    // 测试ARAP
    JsonDict arapDict{
        {"type", JsonNode(std::string("ARAP"))}
    };
    JsonNode arapNode = JsonNode(arapDict);
    
    auto arap = createStrainEnergyDensity<Real>(arapNode);
    EXPECT_NE(arap, nullptr);
    
    // 测试LinearElastic
    JsonDict linearElasticDict{
        {"type", JsonNode(std::string("LinearElastic"))},
        {"mu", JsonNode(5e5)},
        {"lambda", JsonNode(8e5)}
    };
    JsonNode linearElasticNode = JsonNode(linearElasticDict);
}

// 测试错误处理
TEST(FEMDeserializerTest, ErrorHandling) {
    // 测试缺少type字段的Primitive
    JsonDict invalidPrimitiveDict{
        {"mesh", JsonNode(JsonDict{})},
        {"energy", JsonNode(JsonDict{})},
        {"density", JsonNode(1000.0)}
    };
    JsonNode invalidPrimitiveNode = JsonNode(invalidPrimitiveDict);
    
    EXPECT_THROW(deserialize<Primitive>(invalidPrimitiveNode), std::runtime_error);
    
    // 测试未知的Primitive类型
    JsonDict unknownPrimitiveDict{
        {"type", JsonNode(std::string("UnknownPrimitive"))},
        {"mesh", JsonNode(JsonDict{})},
        {"energy", JsonNode(JsonDict{})},
        {"density", JsonNode(1000.0)}
    };
    JsonNode unknownPrimitiveNode = JsonNode(unknownPrimitiveDict);
    
    EXPECT_THROW(deserialize<Primitive>(unknownPrimitiveNode), std::runtime_error);
    
    // 测试未知的StrainEnergyDensity类型
    JsonDict unknownEnergyDict{
        {"type", JsonNode(std::string("UnknownEnergy"))}
    };
    JsonNode unknownEnergyNode = JsonNode(unknownEnergyDict);
    
    EXPECT_THROW(createStrainEnergyDensity<Real>(unknownEnergyNode), std::runtime_error);
}

// 测试新的StrainEnergyDensity工厂函数
TEST(FEMDeserializerTest, StrainEnergyDensityFactoryTest) {
    // 测试StableNeoHookean的工厂方法
    JsonDict stableNeoHookeanDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(1e6)},
        {"lambda", JsonNode(1.5e6)}
    };
    JsonNode stableNeoHookeanNode = JsonNode(stableNeoHookeanDict);
    
    auto energy1 = createStrainEnergyDensity<Real>(stableNeoHookeanNode);
    EXPECT_NE(energy1, nullptr);
    
    // 测试ARAP的工厂方法
    JsonDict arapDict{
        {"type", JsonNode(std::string("ARAP"))}
    };
    JsonNode arapNode = JsonNode(arapDict);
    
    auto energy2 = createStrainEnergyDensity<Real>(arapNode);
    EXPECT_NE(energy2, nullptr);
}

// 测试Linear Solver工厂函数
TEST(FEMDeserializerTest, LinearSolverFactoryTest) {
    // 测试Cholesky Solver
    JsonDict choleskySolverDict{
        {"type", JsonNode(std::string("cholesky-solver"))}
    };
    JsonNode choleskySolverNode = JsonNode(choleskySolverDict);
    
    auto choleskySolver = createLinearSolver(choleskySolverNode);
    EXPECT_NE(choleskySolver, nullptr);
    
    // 测试CG Solver
    JsonDict cgSolverDict{
        {"type", JsonNode(std::string("cg-solver"))},
        {"maxIterations", JsonNode(1000)},
        {"tolerance", JsonNode(1e-6)}
    };
    JsonNode cgSolverNode = JsonNode(cgSolverDict);
    
    auto cgSolver = createLinearSolver(cgSolverNode);
    EXPECT_NE(cgSolver, nullptr);
    
    // 测试CG Solver with preconditioner
    JsonDict cgWithPrecondDict{
        {"type", JsonNode(std::string("cg-solver"))},
        {"maxIterations", JsonNode(500)},
        {"tolerance", JsonNode(1e-8)},
        {"preconditioner", JsonNode(JsonDict{
            {"type", JsonNode(std::string("cholesky-solver"))}
        })}
    };
    JsonNode cgWithPrecondNode = JsonNode(cgWithPrecondDict);
    
    auto cgWithPrecond = createLinearSolver(cgWithPrecondNode);
    EXPECT_NE(cgWithPrecond, nullptr);
}

// 测试Integrator工厂函数
TEST(FEMDeserializerTest, IntegratorFactoryTest) {
    // 首先创建一个简单的系统用于测试
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict energyDict{
        {"type", JsonNode(std::string("ARAP"))}
    };
    
    JsonDict primitiveDict{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    
    JsonDict systemDict{
        {"primitives", JsonNode(JsonList{JsonNode(primitiveDict)})},
        {"colliders", JsonNode(JsonList{})}
    };
    JsonNode systemNode = JsonNode(systemDict);
    
    SystemBuilder systemBuilder;
    auto system = systemBuilder.build(systemNode);
    
    // 测试IPC Implicit Euler Integrator
    JsonDict ipcIntegratorDict{
        {"type", JsonNode(std::string("ipc"))},
        {"config", JsonNode(JsonDict{
            {"type", JsonNode(std::string("implicit-euler"))},
            {"dHat", JsonNode(1e-3)},
            {"eps", JsonNode(1e-2)},
            {"contactStiffness", JsonNode(1e8)},
            {"stepSizeScale", JsonNode(0.8)},
            {"linearSolver", JsonNode(JsonDict{
                {"type", JsonNode(std::string("cholesky-solver"))}
            })}
        })}
    };
    JsonNode ipcIntegratorNode = JsonNode(ipcIntegratorDict);
    
    auto integrator = createIntegrator(system, ipcIntegratorNode);
    EXPECT_NE(integrator, nullptr);
    
    // 测试带有CG solver的integrator
    JsonDict ipcWithCGDict{
        {"type", JsonNode(std::string("ipc"))},
        {"config", JsonNode(JsonDict{
            {"type", JsonNode(std::string("implicit-euler"))},
            {"dHat", JsonNode(5e-4)},
            {"eps", JsonNode(5e-3)},
            {"contactStiffness", JsonNode(5e9)},
            {"stepSizeScale", JsonNode(0.9)},
            {"linearSolver", JsonNode(JsonDict{
                {"type", JsonNode(std::string("cg-solver"))},
                {"maxIterations", JsonNode(2000)},
                {"tolerance", JsonNode(1e-10)}
            })}
        })}
    };
    JsonNode ipcWithCGNode = JsonNode(ipcWithCGDict);
    
    auto integratorWithCG = createIntegrator(system, ipcWithCGNode);
    EXPECT_NE(integratorWithCG, nullptr);
}

// 测试复杂网格反序列化
TEST(FEMDeserializerTest, ComplexMeshDeserialization) {
    // 创建一个立方体网格
    JsonList vertices;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                vertices.push_back(JsonNode(JsonList{
                    JsonNode(static_cast<Real>(i)),
                    JsonNode(static_cast<Real>(j)),
                    JsonNode(static_cast<Real>(k))
                }));
            }
        }
    }
    
    // 创建6个四面体来构成立方体
    JsonList tets;
    // 立方体的6个四面体分解
    std::vector<std::vector<int>> tetIndices = {
        {0, 1, 2, 4}, {1, 2, 4, 6}, {2, 4, 6, 7},
        {1, 2, 3, 6}, {2, 3, 6, 7}, {0, 2, 4, 5}
    };
    
    for (const auto& tet : tetIndices) {
        tets.emplace_back(JsonList{
            JsonNode(tet[0]), JsonNode(tet[1]), 
            JsonNode(tet[2]), JsonNode(tet[3])
        });
    }
    
    JsonDict meshDict{
        {"vertices", JsonNode(vertices)},
        {"tets", JsonNode(tets)}
    };
    
    JsonDict energyDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(1e6)},
        {"lambda", JsonNode(1.5e6)}
    };
    
    JsonDict elasticTetMeshDict{
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    JsonNode elasticTetMeshNode = JsonNode(elasticTetMeshDict);
    
    auto elasticTetMesh = deserialize<ElasticTetMesh>(elasticTetMeshNode);
    
    EXPECT_EQ(elasticTetMesh.mesh.getVertices().size(), 8);
    EXPECT_EQ(elasticTetMesh.mesh.tets.size(), 6);
    EXPECT_DOUBLE_EQ(elasticTetMesh.density, 1000.0);
    
    // 验证表面三角形计算
    auto surfaceView = elasticTetMesh.getSurfaceView();
    EXPECT_GT(surfaceView.size(), 0);
    
    // 验证边计算
    auto edgesView = elasticTetMesh.getEdgesView();
    EXPECT_GT(edgesView.size(), 0);
}

// 测试FEMSimulationBuilder
TEST(FEMDeserializerTest, FEMSimulationBuilderTest) {
    // 创建简单的网格
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict energyDict{
        {"type", JsonNode(std::string("ARAP"))}
    };
    
    JsonDict primitiveDict{
        {"type", JsonNode(std::string("ElasticTetMesh"))},
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    
    JsonDict systemDict{
        {"primitives", JsonNode(JsonList{JsonNode(primitiveDict)})},
        {"colliders", JsonNode(JsonList{})}
    };
    
    JsonDict integratorDict{
        {"type", JsonNode(std::string("ipc"))},
        {"config", JsonNode(JsonDict{
            {"type", JsonNode(std::string("implicit-euler"))},
            {"dHat", JsonNode(1e-3)},
            {"eps", JsonNode(1e-2)},
            {"contactStiffness", JsonNode(1e8)},
            {"stepSizeScale", JsonNode(0.8)},
            {"linearSolver", JsonNode(JsonDict{
                {"type", JsonNode(std::string("cholesky-solver"))}
            })}
        })}
    };
    
    JsonDict simulationDict{
        {"system", JsonNode(systemDict)},
        {"integrator", JsonNode(integratorDict)}
    };
    JsonNode simulationNode = JsonNode(simulationDict);
    
    // 测试完整的FEMSimulation构建
    FEMSimulationBuilder simulationBuilder;
    auto simulation = simulationBuilder.build(simulationNode);
    
    EXPECT_TRUE(simulation.canSimulate());
}

// 测试完整的ElasticTetMesh工作流程
TEST(FEMDeserializerTest, ElasticTetMeshWorkflow) {
    // 1. 通过JSON反序列化创建ElasticTetMesh
    JsonDict meshDict{
        {"vertices", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(1.0), JsonNode(0.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(1.0), JsonNode(0.0)}),
            JsonNode(JsonList{JsonNode(0.0), JsonNode(0.0), JsonNode(1.0)})
        })},
        {"tets", JsonNode(JsonList{
            JsonNode(JsonList{JsonNode(0), JsonNode(1), JsonNode(2), JsonNode(3)})
        })}
    };
    
    JsonDict energyDict{
        {"type", JsonNode(std::string("StableNeoHookean"))},
        {"mu", JsonNode(8e5)},
        {"lambda", JsonNode(1.2e6)}
    };
    
    JsonDict elasticTetMeshDict{
        {"mesh", JsonNode(meshDict)},
        {"energy", JsonNode(energyDict)},
        {"density", JsonNode(1000.0)}
    };
    JsonNode elasticTetMeshNode = JsonNode(elasticTetMeshDict);
    
    auto elasticMesh1 = ElasticTetMesh::static_deserialize(elasticTetMeshNode);
    
    // 2. 通过工厂函数创建相同的ElasticTetMesh
    std::vector<Vector<Real, 3>> vertices = {
        Vector<Real, 3>(0.0, 0.0, 0.0),
        Vector<Real, 3>(1.0, 0.0, 0.0),
        Vector<Real, 3>(0.0, 1.0, 0.0),
        Vector<Real, 3>(0.0, 0.0, 1.0)
    };
    std::vector<Tetrahedron> tets = {Tetrahedron{0, 1, 2, 3}};
    TetMesh mesh{vertices, tets};
    
    auto snhEnergy = std::make_unique<StableNeoHookean<Real>>(8e5, 1.2e6);
    ElasticTetMesh elasticMesh2(mesh, std::move(snhEnergy), 1000.0);
    
    // 3. 验证两种方式创建的对象具有相同的属性
    EXPECT_EQ(elasticMesh1.getVertexCount(), elasticMesh2.getVertexCount());
    EXPECT_EQ(elasticMesh1.mesh.tets.size(), elasticMesh2.mesh.tets.size());
    EXPECT_DOUBLE_EQ(elasticMesh1.density, elasticMesh2.density);
    EXPECT_EQ(elasticMesh1.dofDim(), elasticMesh2.dofDim());
    
    // 4. 验证表面和边缘计算
    auto surfaces1 = elasticMesh1.getSurfaceView();
    auto surfaces2 = elasticMesh2.getSurfaceView();
    EXPECT_EQ(surfaces1.size(), surfaces2.size());
    
    auto edges1 = elasticMesh1.getEdgesView();
    auto edges2 = elasticMesh2.getEdgesView();
    EXPECT_EQ(edges1.size(), edges2.size());
    
    // 5. 验证基本几何属性
    EXPECT_EQ(elasticMesh1.dofDim(), 12); // 4个顶点 * 3个自由度
    EXPECT_GT(surfaces1.size(), 0); // 应该有表面三角形
    EXPECT_GT(edges1.size(), 0);    // 应该有表面边
}
