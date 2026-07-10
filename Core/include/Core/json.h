//
// Created by creeper on 10/24/24.
//

#pragma once
#include <Core/core.h>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace ksk::core {
struct JsonNode;
using JsonDict = std::unordered_map<std::string, JsonNode>;
using JsonList = std::vector<JsonNode>;
struct JsonNode {
  JsonNode() = default;
  explicit JsonNode(std::nullptr_t) : m_value(nullptr) {}
  explicit JsonNode(int value) : m_value(value) {}
  explicit JsonNode(Real value) : m_value(value) {}
  explicit JsonNode(bool value) : m_value(value) {}
  explicit JsonNode(const std::string &value) : m_value(value) {}
  explicit JsonNode(const JsonDict &value) : m_value(value) {}
  explicit JsonNode(const JsonList &value) : m_value(value) {}
  JsonNode(const JsonNode &other) = default;
  JsonNode(JsonNode &&other) noexcept : m_value(std::move(other.m_value)) {
    other.m_value = nullptr;
  }
  JsonNode &operator=(const JsonNode &other) = default;
  JsonNode &operator=(JsonNode &&other) noexcept {
    m_value = std::move(other.m_value);
    other.m_value = nullptr;
    return *this;
  }

  template <typename T> [[nodiscard]] bool is() const {
    return std::holds_alternative<T>(m_value);
  }
  template <typename T> T &as() { return std::get<T>(m_value); }
  template <typename T> const T &as() const { return std::get<T>(m_value); }
  JsonNode &value(const std::string &key) {
    if (!is<JsonDict>())
      throw std::runtime_error("JsonNode is not a dictionary");
    auto &dict = std::get<JsonDict>(m_value);
    if (auto it = dict.find(key); it != dict.end())
      return it->second;
    throw std::runtime_error("Key not found");
  }
  [[nodiscard]] const JsonNode &value(const std::string &key) const {
    if (!is<JsonDict>())
      throw std::runtime_error("JsonNode is not a dictionary");
    auto &dict = std::get<JsonDict>(m_value);
    if (const auto it = dict.find(key); it != dict.end())
      return it->second;
    throw std::runtime_error("Key not found");
  }

private:
  std::variant<std::nullptr_t, int, Real, bool, std::string, JsonDict, JsonList>
      m_value{};
};

std::optional<JsonNode> parseJson(std::string_view json) noexcept;
std::optional<JsonNode>
loadJsonFile(const std::filesystem::path &path) noexcept;
} // namespace core
