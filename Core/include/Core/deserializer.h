//
// Created by creeper on 10/31/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <Core/json.h>
#include <Core/reflection.h>

namespace ksk::core {
template<typename T>
struct custom_deserializer {
  static T do_deserialize(const JsonNode &node) {
    throw std::runtime_error("Custom deserializer is not implemented for type " + std::string(typeid(T).name()));
  }
};

template <typename T>
T basic_deserialize_impl(const JsonNode& node) {
  if (!node.is<T>())
    throw std::runtime_error("Invalid node type for basic deserialization");
  return node.as<T>();
}

template<typename T>
T deserialize(const JsonNode& node) {
  if constexpr (std::is_same_v<T, std::string>)
    return basic_deserialize_impl<std::string>(node);
  else if constexpr (std::is_same_v<T, int>)
    return basic_deserialize_impl<int>(node);
  else if constexpr (std::is_same_v<T, Real>)
    return basic_deserialize_impl<Real>(node);
  else if constexpr (std::is_same_v<T, bool>)
    return basic_deserialize_impl<bool>(node);
  else if constexpr (can_reflect<T>()) {
    T obj;
    if (!node.is<JsonDict>())
      throw std::runtime_error("Expected a dictionary for deserializing reflectable type");
    const auto& dict = node.as<JsonDict>();

    obj.forEachMember([&](auto name, auto& member) {
        using member_type = typename std::decay_t<decltype(member)>;
        member = deserialize<member_type>(dict.at(name));
    });
    return obj;
  } else {
    if constexpr (requires { T::static_deserialize(node); })
      return T::static_deserialize(node);
    else
      return custom_deserializer<T>::do_deserialize(node);
  }
}

template<typename T>
struct custom_deserializer<std::vector<T> > {
  static std::vector<T> do_deserialize(const JsonNode &node) {
    if (!node.is<JsonList>())
      throw std::runtime_error("Expected a list");
    const auto& list = node.as<JsonList>();
    std::vector<T> ret;
    ret.reserve(list.size());
    for (const auto& item : list)
      ret.emplace_back(deserialize<T>(item));
    return ret;
  }
};

template<typename T, size_t N>
struct custom_deserializer<std::array<T, N> > {
  static std::array<T, N> do_deserialize(const JsonNode &node) {
    if (!node.is<JsonList>())
      throw std::runtime_error("Expected a list");
    const auto& list = node.as<JsonList>();
    if (N == list.size()) {
      std::array<T, N> ret;
      for (size_t i = 0; i < list.size(); i++)
        ret[i] = deserialize<T>(list[i]);
      return ret;
    }
    throw std::runtime_error("Expected a list of size " + std::to_string(N));
  }
};

template<typename T, int N>
struct custom_deserializer<glm::vec<N, T>> {
  static glm::vec<N, T> do_deserialize(const JsonNode &node) {
    if (!node.is<JsonList>())
      throw std::runtime_error("Expected a list for glm vector deserialization");
    const auto& list = node.as<JsonList>();
    if (N == list.size()) {
      glm::vec<N, T> ret;
      for (int i = 0; i < N; i++)
        ret[i] = deserialize<T>(list[i]);
      return ret;
    }
    throw std::runtime_error("Expected a list of size " + std::to_string(N));
  }
};
}


