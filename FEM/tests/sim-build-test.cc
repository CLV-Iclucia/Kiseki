//
// Created by CreeperIclucia-Vader on 25-5-29.
//

#include <gtest/gtest.h>
#include <Core/json.h>
#include <fem/fem-simulation.h>
#include <fem/system.h>
#include <Deform/strain-energy-density.h>
#include <fstream>
#include <filesystem>

using namespace ksk::fem;

class FEMSimulationBuildTest : public ::testing::Test {
protected:
    // 从JSON字符串构建FEMSimulation
    FEMSimulation buildFromJsonString(const std::string& jsonStr) {
        auto jsonNode = ksk::core::parseJson(jsonStr);
        if (!jsonNode || !jsonNode->is<ksk::core::JsonDict>()) {
            throw std::runtime_error("Json Parse failed");
        }
        FEMSimulationBuilder builder;
        return builder.build(*jsonNode);
    }
    
    // 从JSON文件构建FEMSimulation
    FEMSimulation buildFromJsonFile(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }
        
        std::string jsonStr((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        return buildFromJsonString(jsonStr);
    }
    
    std::string getTestScenarioPath(const std::string& filename) {
        std::filesystem::path testDir = std::filesystem::current_path() / "FEM" / "tests" / "test-scenarios";
        if (!std::filesystem::exists(testDir)) {
            testDir = std::filesystem::current_path() / "tests" / "test-scenarios";
        }
        if (!std::filesystem::exists(testDir)) {
            testDir = std::filesystem::current_path() / "test-scenarios";
        }
        
        return (testDir / filename).string();
    }
};

// 测试基础的单四面体ARAP能量仿真构建
TEST_F(FEMSimulationBuildTest, BasicARAPSimulationFromString) {
    std::string jsonConfig = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "tets": [[0, 1, 2, 3]]
                },
                "energy": {
                    "type": "ARAP"
                },
                "density": 1000.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "dHat": 1e-3,
                "eps": 1e-2,
                "contactStiffness": 1e8,
                "stepSizeScale": 0.8,
                "linearSolver": {
                    "type": "cholesky-solver"
                }
            }
        }
    })";
    
    auto simulation = buildFromJsonString(jsonConfig);
    EXPECT_TRUE(simulation.canSimulate());
}

// 测试StableNeoHookean能量与CG求解器
TEST_F(FEMSimulationBuildTest, StableNeoHookeanWithCGSolverFromString) {
    std::string jsonConfig = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "tets": [[0, 1, 2, 3]]
                },
                "energy": {
                    "type": "StableNeoHookean",
                    "mu": 8e5,
                    "lambda": 1.2e6
                },
                "density": 1200.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "dHat": 5e-4,
                "eps": 5e-3,
                "contactStiffness": 5e9,
                "stepSizeScale": 0.9,
                "linearSolver": {
                    "type": "cg-solver",
                    "maxIterations": 1000,
                    "tolerance": 1e-8
                }
            }
        }
    })";
    
    auto simulation = buildFromJsonString(jsonConfig);
    EXPECT_TRUE(simulation.canSimulate());
}

TEST_F(FEMSimulationBuildTest, LinearElasticWithPreconditionerFromString) {
    std::string jsonConfig = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [
                        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
                    ],
                    "tets": [
                        [0, 1, 2, 4], [1, 2, 4, 6], [2, 4, 6, 7],
                        [1, 2, 3, 6], [2, 3, 6, 7], [0, 2, 4, 5]
                    ]
                },
                "energy": {
                    "type": "LinearElastic",
                    "mu": 5e5,
                    "lambda": 8e5
                },
                "density": 800.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "dHat": 1e-3,
                "eps": 1e-2,
                "contactStiffness": 1e8,
                "stepSizeScale": 0.8,
                "linearSolver": {
                    "type": "cg-solver",
                    "maxIterations": 2000,
                    "tolerance": 1e-10,
                    "preconditioner": {
                        "type": "cholesky-solver"
                    }
                }
            }
        }
    })";
    
    auto simulation = buildFromJsonString(jsonConfig);
    EXPECT_TRUE(simulation.canSimulate());
}

// 测试多primitive系统
TEST_F(FEMSimulationBuildTest, MultiPrimitiveSystemFromString) {
    std::string jsonConfig = R"(
    {
        "system": {
            "primitives": [
                {
                    "type": "ElasticTetMesh",
                    "mesh": {
                        "vertices": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]
                        ],
                        "tets": [[0, 1, 2, 3]]
                    },
                    "energy": {
                        "type": "ARAP"
                    },
                    "density": 1000.0
                },
                {
                    "type": "ElasticTetMesh",
                    "mesh": {
                        "vertices": [
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                            [2.0, 0.0, 1.0]
                        ],
                        "tets": [[0, 1, 2, 3]]
                    },
                    "energy": {
                        "type": "StableNeoHookean",
                        "mu": 6e5,
                        "lambda": 9e5
                    },
                    "density": 1100.0
                }
            ],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "dHat": 1e-3,
                "eps": 1e-2,
                "contactStiffness": 1e8,
                "stepSizeScale": 0.8,
                "linearSolver": {
                    "type": "cholesky-solver"
                }
            }
        }
    })";
    
    auto simulation = buildFromJsonString(jsonConfig);
    EXPECT_TRUE(simulation.canSimulate());
}

// 测试从文件加载基础ARAP场景
TEST_F(FEMSimulationBuildTest, LoadBasicARAPFromFile) {
    std::string filePath = getTestScenarioPath("basic-arap.json");
    
    // 只有在文件存在时才测试
    if (std::filesystem::exists(filePath)) {
        auto simulation = buildFromJsonFile(filePath);
        EXPECT_TRUE(simulation.canSimulate());
    } else {
        GTEST_SKIP() << "Test scenario file not found: " << filePath;
    }
}

// 测试从文件加载StableNeoHookean场景
TEST_F(FEMSimulationBuildTest, LoadStableNeoHookeanFromFile) {
    std::string filePath = getTestScenarioPath("stable-neohookean-cg.json");
    
    if (std::filesystem::exists(filePath)) {
        auto simulation = buildFromJsonFile(filePath);
        EXPECT_TRUE(simulation.canSimulate());
    } else {
        GTEST_SKIP() << "Test scenario file not found: " << filePath;
    }
}

// 测试从文件加载复杂预条件子场景
TEST_F(FEMSimulationBuildTest, LoadComplexWithPreconditionerFromFile) {
    std::string filePath = getTestScenarioPath("complex-with-preconditioner.json");
    
    if (std::filesystem::exists(filePath)) {
        auto simulation = buildFromJsonFile(filePath);
        EXPECT_TRUE(simulation.canSimulate());
    } else {
        GTEST_SKIP() << "Test scenario file not found: " << filePath;
    }
}

// 测试从文件加载多primitive场景
TEST_F(FEMSimulationBuildTest, LoadMultiPrimitiveFromFile) {
    std::string filePath = getTestScenarioPath("multi-primitive.json");
    
    if (std::filesystem::exists(filePath)) {
        auto simulation = buildFromJsonFile(filePath);
        EXPECT_TRUE(simulation.canSimulate());
    } else {
        GTEST_SKIP() << "Test scenario file not found: " << filePath;
    }
}

// 测试错误处理
TEST_F(FEMSimulationBuildTest, ErrorHandlingFromString) {
    // 测试缺少system字段
    std::string invalidConfig1 = R"(
    {
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler"
            }
        }
    })";
    
    EXPECT_THROW(buildFromJsonString(invalidConfig1), std::runtime_error);
    
    // 测试缺少integrator字段
    std::string invalidConfig2 = R"(
    {
        "system": {
            "primitives": [],
            "colliders": []
        }
    })";
    
    EXPECT_THROW(buildFromJsonString(invalidConfig2), std::runtime_error);
    
    // 测试无效的能量类型
    std::string invalidConfig3 = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "tets": [[0, 1, 2, 3]]
                },
                "energy": {
                    "type": "UnknownEnergy"
                },
                "density": 1000.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "linearSolver": {"type": "cholesky-solver"}
            }
        }
    })";
    
    EXPECT_THROW(buildFromJsonString(invalidConfig3), std::runtime_error);
    
    // 测试无效的求解器类型
    std::string invalidConfig4 = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "tets": [[0, 1, 2, 3]]
                },
                "energy": {"type": "ARAP"},
                "density": 1000.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "linearSolver": {"type": "unknown-solver"}
            }
        }
    })";
    
    EXPECT_THROW(buildFromJsonString(invalidConfig4), std::runtime_error);
}

// 测试JSON解析错误
TEST_F(FEMSimulationBuildTest, JsonParsingErrorFromString) {
    // 无效的JSON语法
    std::string invalidJson = R"(
    {
        "system": {
            "primitives": [
                "type": "ElasticTetMesh"  // 缺少大括号
            ]
        }
    })";
    
    EXPECT_THROW(buildFromJsonString(invalidJson), std::runtime_error);
}

// 测试文件读取错误
TEST_F(FEMSimulationBuildTest, FileReadingError) {
    // 测试不存在的文件
    EXPECT_THROW(buildFromJsonFile("non-existent-file.json"), std::runtime_error);
}

// 测试完整配置的一致性
TEST_F(FEMSimulationBuildTest, CompleteConfigurationConsistencyFromString) {
    std::string jsonConfig = R"(
    {
        "system": {
            "primitives": [{
                "type": "ElasticTetMesh",
                "mesh": {
                    "vertices": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ],
                    "tets": [[0, 1, 2, 3]]
                },
                "energy": {
                    "type": "StableNeoHookean",
                    "mu": 7e5,
                    "lambda": 1.1e6
                },
                "density": 950.0
            }],
            "colliders": []
        },
        "integrator": {
            "type": "ipc",
            "config": {
                "type": "implicit-euler",
                "dHat": 8e-4,
                "eps": 8e-3,
                "contactStiffness": 2e8,
                "stepSizeScale": 0.85,
                "linearSolver": {
                    "type": "cg-solver",
                    "maxIterations": 1200,
                    "tolerance": 5e-9
                }
            }
        }
    })";
    
    // 构建两次相同的仿真
    auto simulation1 = buildFromJsonString(jsonConfig);
    auto simulation2 = buildFromJsonString(jsonConfig);
    
    // 两个仿真都应该能正常工作
    EXPECT_TRUE(simulation1.canSimulate());
    EXPECT_TRUE(simulation2.canSimulate());
}
