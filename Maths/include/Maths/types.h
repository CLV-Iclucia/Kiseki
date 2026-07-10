//
// Created by creeper on 5/23/24.
//

#pragma once
#include <Core/deserializer.h>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
namespace ksk::maths {
using Eigen::Dynamic;
template <typename T, int Rows, int Cols>
using Matrix = Eigen::Matrix<T, Rows, Cols>;
template <typename T, int Dim>
using Vector = Eigen::Matrix<T, Dim, 1>;
template <typename T, int Dim>
using VecView = Eigen::Map<Vector<T, Dim>>;
template <typename T, int Dim>
using CVecView = Eigen::Map<const Vector<T, Dim>>;
template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
using Real = double;
}


template <typename T, int N>
struct ksk::core::custom_deserializer<ksk::maths::Vector<T, N>> {
  static maths::Vector<T, N> do_deserialize(const JsonNode &node) {
    if (!node.is<JsonList>())
      throw std::runtime_error("Expected a list");

    const auto& list = node.as<JsonList>();
    if (N == Eigen::Dynamic || N == list.size()) {
      maths::Vector<T, N> ret;
      for (int i = 0; i < list.size(); i++)
        ret[i] = list[i].as<T>();
      return ret;
    } else
      throw std::runtime_error("Expected a list of size " + std::to_string(N));
  }
};
