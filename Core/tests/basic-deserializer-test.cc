//
// Created by CreeperIclucia-Vader on 25-5-26.
//

#include <Core/deserializer.h>
#include <Core/json.h>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <typeindex>
#include <variant>
#include <vector>

using namespace ksk::core;

// 测试基本类型反序列化
TEST(BasicDeserializerTest, BasicTypes) {
  JsonNode strNode = JsonNode(std::string("hello"));
  auto strResult = deserialize<std::string>(strNode);
  EXPECT_EQ(strResult, "hello");

  JsonNode intNode = JsonNode(42);
  auto intResult = deserialize<int>(intNode);
  EXPECT_EQ(intResult, 42);

  JsonNode realNode = JsonNode(3.14);
  auto realResult = deserialize<Real>(realNode);
  EXPECT_DOUBLE_EQ(realResult, 3.14);

  JsonNode boolNode = JsonNode(true);
  auto boolResult = deserialize<bool>(boolNode);
  EXPECT_TRUE(boolResult);
}

// 测试容器类型反序列化
TEST(BasicDeserializerTest, ContainerTypes) {
  JsonList vecList{JsonNode(1), JsonNode(2), JsonNode(3), JsonNode(4),
                   JsonNode(5)};
  JsonNode vecNode = JsonNode(vecList);
  auto vecResult = deserialize<std::vector<int>>(vecNode);
  EXPECT_EQ(vecResult.size(), 5);
  EXPECT_EQ(vecResult[0], 1);
  EXPECT_EQ(vecResult[4], 5);

  JsonList arrList{JsonNode(1.0), JsonNode(2.0), JsonNode(3.0)};
  JsonNode arrNode = JsonNode(arrList);
  auto arrResult = deserialize<std::array<Real, 3>>(arrNode);
  EXPECT_DOUBLE_EQ(arrResult[0], 1.0);
  EXPECT_DOUBLE_EQ(arrResult[1], 2.0);
  EXPECT_DOUBLE_EQ(arrResult[2], 3.0);

  JsonList strList{JsonNode(std::string("hello")),
                   JsonNode(std::string("world"))};
  JsonNode strVecNode = JsonNode(strList);
  auto strVecResult = deserialize<std::vector<std::string>>(strVecNode);
  EXPECT_EQ(strVecResult.size(), 2);
  EXPECT_EQ(strVecResult[0], "hello");
  EXPECT_EQ(strVecResult[1], "world");
}

// 测试自定义类型反序列化 - 使用不同的反序列化方式

// 1. 使用静态方法反序列化
struct StaticDeserializeType {
  int value;
  std::string name;

  static StaticDeserializeType static_deserialize(const JsonNode &node) {
    if (!node.is<JsonDict>())
      throw std::runtime_error("Expected a dictionary");
    const auto &dict = node.as<JsonDict>();
    return StaticDeserializeType{dict.at("value").as<int>(),
                                 dict.at("name").as<std::string>()};
  }
};

// 2. 使用自定义反序列化器
struct CustomDeserializeType {
  std::vector<int> values;
  std::map<std::string, double> metrics;
};

template <> struct ksk::core::custom_deserializer<CustomDeserializeType> {
  static CustomDeserializeType do_deserialize(const JsonNode &node) {
    if (!node.is<JsonDict>())
      throw std::runtime_error("Expected a dictionary");
    const auto &dict = node.as<JsonDict>();

    CustomDeserializeType result;
    result.values = deserialize<std::vector<int>>(dict.at("values"));

    const auto &metrics = dict.at("metrics").as<JsonDict>();
    for (const auto &[key, value] : metrics) {
      result.metrics[key] = value.as<Real>();
    }
    return result;
  }
};

struct ReflectedType {
  int id;
  std::string name;
  std::vector<double> scores;
  bool active;
  REFLECT(id, name, scores, active)
  static_assert(can_reflect<ReflectedType>());
};

// 4. 复杂嵌套类型
struct ComplexType {
  struct InnerType {
    int x;
    std::string y;

    static InnerType static_deserialize(const JsonNode &node) {
      if (!node.is<JsonDict>())
        throw std::runtime_error("Expected a dictionary");
      const auto &dict = node.as<JsonDict>();
      return InnerType{dict.at("x").as<int>(), dict.at("y").as<std::string>()};
    }
  };

  std::vector<InnerType> items;
  std::variant<int, std::string> value;
  std::optional<double> score;

  static ComplexType static_deserialize(const JsonNode &node) {
    if (!node.is<JsonDict>())
      throw std::runtime_error("Expected a dictionary");
    const auto &dict = node.as<JsonDict>();

    ComplexType result;
    result.items = deserialize<std::vector<InnerType>>(dict.at("items"));

    const auto &valueNode = dict.at("value");
    if (valueNode.is<int>()) {
      result.value = valueNode.as<int>();
    } else if (valueNode.is<std::string>()) {
      result.value = valueNode.as<std::string>();
    } else {
      throw std::runtime_error("Invalid value type");
    }

    if (dict.contains("score")) {
      result.score = dict.at("score").as<Real>();
    }

    return result;
  }
};

TEST(BasicDeserializerTest, CustomTypes) {
  // 测试静态方法反序列化
  JsonDict staticDict{{"value", JsonNode(42)},
                      {"name", JsonNode(std::string("test"))}};
  JsonNode staticNode = JsonNode(staticDict);
  auto staticResult = deserialize<StaticDeserializeType>(staticNode);
  EXPECT_EQ(staticResult.value, 42);
  EXPECT_EQ(staticResult.name, "test");

  // 测试自定义反序列化器
  JsonDict customDict{
      {"values", JsonNode(JsonList{JsonNode(1), JsonNode(2), JsonNode(3)})},
      {"metrics", JsonNode(JsonDict{{"metric1", JsonNode(1.5)},
                                    {"metric2", JsonNode(2.5)}})}};
  JsonNode customNode = JsonNode(customDict);
  auto customResult = deserialize<CustomDeserializeType>(customNode);
  EXPECT_EQ(customResult.values.size(), 3);
  EXPECT_EQ(customResult.metrics.size(), 2);
  EXPECT_DOUBLE_EQ(customResult.metrics["metric1"], 1.5);
  EXPECT_DOUBLE_EQ(customResult.metrics["metric2"], 2.5);

  // 测试反射机制
  JsonDict reflectedDict{
      {"id", JsonNode(1)},
      {"name", JsonNode(std::string("reflected"))},
      {"scores", JsonNode(JsonList{JsonNode(1.0), JsonNode(2.0)})},
      {"active", JsonNode(true)}};
  JsonNode reflectedNode = JsonNode(reflectedDict);
  auto reflectedResult = deserialize<ReflectedType>(reflectedNode);
  EXPECT_EQ(reflectedResult.id, 1);
  EXPECT_EQ(reflectedResult.name, "reflected");
  EXPECT_EQ(reflectedResult.scores.size(), 2);
  EXPECT_TRUE(reflectedResult.active);

  // 测试复杂嵌套类型
  JsonDict complexDict{
      {"items",
       JsonNode(JsonList{
           JsonNode(JsonDict{{"x", JsonNode(1)},
                             {"y", JsonNode(std::string("item1"))}}),
           JsonNode(JsonDict{{"x", JsonNode(2)},
                             {"y", JsonNode(std::string("item2"))}})})},
      {"value", JsonNode(std::string("string_value"))},
      {"score", JsonNode(3.14)}};
  JsonNode complexNode = JsonNode(complexDict);
  auto complexResult = deserialize<ComplexType>(complexNode);
  EXPECT_EQ(complexResult.items.size(), 2);
  EXPECT_EQ(complexResult.items[0].x, 1);
  EXPECT_EQ(complexResult.items[0].y, "item1");
  EXPECT_EQ(complexResult.items[1].x, 2);
  EXPECT_EQ(complexResult.items[1].y, "item2");
  EXPECT_TRUE(std::holds_alternative<std::string>(complexResult.value));
  EXPECT_EQ(std::get<std::string>(complexResult.value), "string_value");
  EXPECT_TRUE(complexResult.score.has_value());
  EXPECT_DOUBLE_EQ(*complexResult.score, 3.14);
}

// 测试错误处理
TEST(BasicDeserializerTest, ErrorHandling) {
  // 无效类型
  JsonNode invalidNode = JsonNode(std::string("not a number"));
  EXPECT_THROW(deserialize<int>(invalidNode), std::runtime_error);

  // 错误的数组大小
  JsonList wrongSizeList{JsonNode(1), JsonNode(2)};
  JsonNode wrongSizeNode = JsonNode(wrongSizeList);
  try {
    deserialize<std::array<int, 3>>(wrongSizeNode);
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_STREQ(e.what(), "Expected a list of size 3");
  }

  // 无效的变体类型
  JsonDict invalidVariantDict{{"items", JsonNode(JsonList{})},
                              {"value", JsonNode(JsonList{})}, // 无效的变体类型
                              {"score", JsonNode(3.14)}};
  JsonNode invalidVariantNode = JsonNode(invalidVariantDict);
  EXPECT_THROW(deserialize<ComplexType>(invalidVariantNode),
               std::runtime_error);
}