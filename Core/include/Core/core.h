//
// Created by creeper on 23-8-13.
//

#pragma once
#include <glm/glm.hpp>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#endif
namespace ksk::core {
using Real = double;
using Index = int;
using Vec2i = glm::ivec2;
using Vec3i = glm::ivec3;
using Vec4i = glm::ivec4;
using Vec2u = glm::uvec2;
using Vec3u = glm::uvec3;
using Vec4u = glm::uvec4;
using Vec2f = glm::vec2;
using Vec3f = glm::vec3;
using Vec4f = glm::vec4;
using Vec2d = glm::dvec2;
using Vec3d = glm::dvec3;
using Vec4d = glm::dvec4;
using Mat2f = glm::mat2;
using Mat3f = glm::mat3;
using Mat4f = glm::mat4;
using Mat2d = glm::dmat2;
using Mat3d = glm::dmat3;
using Mat4d = glm::dmat4;
using uint = unsigned int;
template <typename T, int Dim> struct TVector {
  static_assert(Dim == 2 || Dim == 3 || Dim == 4, "Dim must be 2, 3 or 4");
};
template <typename T> struct TVector<T, 2> {
  using type = glm::tvec2<T>;
};
template <typename T> struct TVector<T, 3> {
  using type = glm::tvec3<T>;
};
template <typename T> struct TVector<T, 4> {
  using type = glm::tvec4<T>;
};
template <typename T, int Dim> using Vector = typename TVector<T, Dim>::type;
template <typename T, int Dim> struct TMatrix {
  static_assert(Dim == 2 || Dim == 3 || Dim == 4, "Dim must be 2, 3 or 4");
};
template <typename T> struct TMatrix<T, 2> {
  using type = glm::tmat2x2<T>;
};
template <typename T> struct TMatrix<T, 3> {
  using type = glm::tmat3x3<T>;
};
template <typename T> struct TMatrix<T, 4> {
  using type = glm::tmat4x4<T>;
};
template <typename T, int Dim> using Matrix = typename TMatrix<T, Dim>::type;
} // namespace ksk::core
