//
// Created by CreeperIclucia-Vader on 25-5-26.
//

#include <gtest/gtest.h>
#include <Core/json.h>
#include <sstream>

using namespace ksk::core;

TEST(JsonParserTest, BasicTypes) {
    // 测试基本类型解析
    auto result = parseJson(R"({
        "int": 42,
        "real": 3.14159,
        "bool": true,
        "string": "hello world"
    })");

    ASSERT_TRUE(result.has_value());
    auto& node = result.value();
    
    EXPECT_EQ(node.value("int").as<int>(), 42);
    EXPECT_DOUBLE_EQ(node.value("real").as<Real>(), 3.14159);
    EXPECT_TRUE(node.value("bool").as<bool>());
    EXPECT_EQ(node.value("string").as<std::string>(), "hello world");
}

TEST(JsonParserTest, Arrays) {
    // 测试数组解析
    auto result = parseJson(R"({
        "empty_array": [],
        "int_array": [1, 2, 3, 4, 5],
        "mixed_array": [1, "two", 3.0, true],
        "nested_array": [[1, 2], [3, 4], [5, 6]]
    })");

    ASSERT_TRUE(result.has_value());
    auto& node = result.value();
    
    const auto& emptyArray = node.value("empty_array").as<JsonList>();
    EXPECT_TRUE(emptyArray.empty());

    const auto& intArray = node.value("int_array").as<JsonList>();
    EXPECT_EQ(intArray.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(intArray[i].as<int>(), i + 1);
    }

    const auto& mixedArray = node.value("mixed_array").as<JsonList>();
    EXPECT_EQ(mixedArray.size(), 4);
    EXPECT_EQ(mixedArray[0].as<int>(), 1);
    EXPECT_EQ(mixedArray[1].as<std::string>(), "two");
    EXPECT_DOUBLE_EQ(mixedArray[2].as<Real>(), 3.0);
    EXPECT_TRUE(mixedArray[3].as<bool>());

    const auto& nestedArray = node.value("nested_array").as<JsonList>();
    EXPECT_EQ(nestedArray.size(), 3);
    for (int i = 0; i < 3; ++i) {
        const auto& innerArray = nestedArray[i].as<JsonList>();
        EXPECT_EQ(innerArray.size(), 2);
        EXPECT_EQ(innerArray[0].as<int>(), i * 2 + 1);
        EXPECT_EQ(innerArray[1].as<int>(), i * 2 + 2);
    }
}

TEST(JsonParserTest, Objects) {
    // 测试对象解析
    auto result = parseJson(R"({
        "empty_object": {},
        "simple_object": {
            "name": "test",
            "value": 42
        },
        "nested_object": {
            "level1": {
                "level2": {
                    "level3": "deep"
                }
            }
        },
        "complex_object": {
            "array": [1, 2, 3],
            "object": {
                "key": "value"
            },
            "mixed": [1, {"nested": "value"}, 3]
        }
    })");

    ASSERT_TRUE(result.has_value());
    auto& node = result.value();
    
    const auto& emptyObject = node.value("empty_object").as<JsonDict>();
    EXPECT_TRUE(emptyObject.empty());

    const auto& simpleObject = node.value("simple_object").as<JsonDict>();
    EXPECT_EQ(simpleObject.at("name").as<std::string>(), "test");
    EXPECT_EQ(simpleObject.at("value").as<int>(), 42);

    const auto& nestedObject = node.value("nested_object").as<JsonDict>();
    EXPECT_EQ(nestedObject.at("level1").as<JsonDict>()
                     .at("level2").as<JsonDict>()
                     .at("level3").as<std::string>(), "deep");

    const auto& complexObject = node.value("complex_object").as<JsonDict>();
    const auto& array = complexObject.at("array").as<JsonList>();
    EXPECT_EQ(array.size(), 3);
    EXPECT_EQ(array[0].as<int>(), 1);
    
    const auto& object = complexObject.at("object").as<JsonDict>();
    EXPECT_EQ(object.at("key").as<std::string>(), "value");
    
    const auto& mixed = complexObject.at("mixed").as<JsonList>();
    EXPECT_EQ(mixed.size(), 3);
    EXPECT_EQ(mixed[0].as<int>(), 1);
    EXPECT_EQ(mixed[1].as<JsonDict>().at("nested").as<std::string>(), "value");
    EXPECT_EQ(mixed[2].as<int>(), 3);
}

TEST(JsonParserTest, SpecialValues) {
    // 测试特殊值解析
    auto result = parseJson(R"({
        "scientific": 1.23e-4,
        "escaped": "line1\nline2\t\"quoted\"",
        "whitespace": "  spaces  "
    })");

    ASSERT_TRUE(result.has_value());
    auto& node = result.value();
    
    EXPECT_DOUBLE_EQ(node.value("scientific").as<Real>(), 1.23e-4);
    EXPECT_EQ(node.value("escaped").as<std::string>(), "line1\nline2\t\"quoted\"");
    EXPECT_EQ(node.value("whitespace").as<std::string>(), "  spaces  ");
}

TEST(JsonParserTest, InvalidJson) {
    EXPECT_FALSE(parseJson("{").has_value());  // 未闭合的对象
    EXPECT_FALSE(parseJson("}").has_value());  // 未开始的对象
    EXPECT_FALSE(parseJson("[").has_value());  // 未闭合的数组
    EXPECT_FALSE(parseJson("]").has_value());  // 未开始的数组
    EXPECT_FALSE(parseJson("{\"key\":}").has_value());  // 缺少值
    EXPECT_FALSE(parseJson("{\"key\":,}").has_value());  // 无效的逗号
    EXPECT_FALSE(parseJson("{\"key\":\"value\",}").has_value());  // 多余的逗号
    EXPECT_FALSE(parseJson("[1,2,3,]").has_value());  // 数组末尾多余的逗号
    EXPECT_FALSE(parseJson("{\"key\":\"value\" \"key2\":\"value2\"}").has_value());  // 缺少逗号
    EXPECT_FALSE(parseJson("[1 2 3]").has_value());  // 数组元素间缺少逗号
    EXPECT_FALSE(parseJson("{\"key\":\"value\"} extra").has_value());  // 额外的内容
    EXPECT_FALSE(parseJson("{\"key\":\"value\"},").has_value());  // 对象后的逗号
    EXPECT_FALSE(parseJson("[1,2,3],").has_value());  // 数组后的逗号
    EXPECT_FALSE(parseJson("{\"key\":\"value\"} [1,2,3]").has_value());  // 多个根元素
}

TEST(JsonParserTest, ComplexNested) {
    // 测试复杂的嵌套结构
    auto result = parseJson(R"({
        "simulation": {
            "name": "complex_test",
            "version": 1.0,
            "parameters": {
                "gravity": [0.0, -9.81, 0.0],
                "timestep": 0.001,
                "iterations": 100
            },
            "materials": [
                {
                    "name": "material1",
                    "density": 1000.0,
                    "elasticity": {
                        "young": 1e6,
                        "poisson": 0.3
                    }
                },
                {
                    "name": "material2",
                    "density": 2000.0,
                    "elasticity": {
                        "young": 2e6,
                        "poisson": 0.4
                    }
                }
            ],
            "boundary_conditions": {
                "fixed": [0, 1, 2],
                "loads": [
                    {
                        "node": 10,
                        "force": [0.0, -100.0, 0.0]
                    },
                    {
                        "node": 11,
                        "force": [50.0, 0.0, 0.0]
                    }
                ]
            }
        }
    })");

    ASSERT_TRUE(result.has_value());
    auto& node = result.value();
    auto& sim = node.value("simulation").as<JsonDict>();
    
    // 检查基本信息
    EXPECT_EQ(sim.at("name").as<std::string>(), "complex_test");
    EXPECT_DOUBLE_EQ(sim.at("version").as<Real>(), 1.0);
    
    // 检查参数
    auto& params = sim.at("parameters").as<JsonDict>();
    auto& gravity = params.at("gravity").as<JsonList>();
    EXPECT_DOUBLE_EQ(gravity[0].as<Real>(), 0.0);
    EXPECT_DOUBLE_EQ(gravity[1].as<Real>(), -9.81);
    EXPECT_DOUBLE_EQ(gravity[2].as<Real>(), 0.0);
    EXPECT_DOUBLE_EQ(params.at("timestep").as<Real>(), 0.001);
    EXPECT_EQ(params.at("iterations").as<int>(), 100);
    
    // 检查材料
    auto& materials = sim.at("materials").as<JsonList>();
    EXPECT_EQ(materials.size(), 2);
    
    auto& mat1 = materials[0].as<JsonDict>();
    EXPECT_EQ(mat1.at("name").as<std::string>(), "material1");
    EXPECT_DOUBLE_EQ(mat1.at("density").as<Real>(), 1000.0);
    auto& elas1 = mat1.at("elasticity").as<JsonDict>();
    EXPECT_DOUBLE_EQ(elas1.at("young").as<Real>(), 1e6);
    EXPECT_DOUBLE_EQ(elas1.at("poisson").as<Real>(), 0.3);
    
    // 检查边界条件
    auto& bc = sim.at("boundary_conditions").as<JsonDict>();
    auto& fixed = bc.at("fixed").as<JsonList>();
    EXPECT_EQ(fixed.size(), 3);
    EXPECT_EQ(fixed[0].as<int>(), 0);
    EXPECT_EQ(fixed[1].as<int>(), 1);
    EXPECT_EQ(fixed[2].as<int>(), 2);
    
    auto& loads = bc.at("loads").as<JsonList>();
    EXPECT_EQ(loads.size(), 2);
    
    auto& load1 = loads[0].as<JsonDict>();
    EXPECT_EQ(load1.at("node").as<int>(), 10);
    auto& force1 = load1.at("force").as<JsonList>();
    EXPECT_DOUBLE_EQ(force1[0].as<Real>(), 0.0);
    EXPECT_DOUBLE_EQ(force1[1].as<Real>(), -100.0);
    EXPECT_DOUBLE_EQ(force1[2].as<Real>(), 0.0);
}
