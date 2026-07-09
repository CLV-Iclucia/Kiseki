#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <span>
#include <type_traits>

namespace sim::rhi {

namespace detail {

template <class T>
struct IsDirectStructuredData : std::bool_constant<
    std::is_arithmetic_v<T> && std::is_trivially_copyable_v<T>> {};

template <glm::length_t L, class T, glm::qualifier Q>
struct IsDirectStructuredData<glm::vec<L, T, Q>>
    : std::bool_constant<
          std::is_arithmetic_v<T> &&
          std::is_trivially_copyable_v<glm::vec<L, T, Q>> &&
          sizeof(glm::vec<L, T, Q>) == static_cast<size_t>(L) * sizeof(T)> {};

template <glm::length_t C, glm::length_t R, class T, glm::qualifier Q>
struct IsDirectStructuredData<glm::mat<C, R, T, Q>>
    : std::bool_constant<
          std::is_arithmetic_v<T> &&
          std::is_trivially_copyable_v<glm::mat<C, R, T, Q>> &&
          sizeof(glm::mat<C, R, T, Q>) ==
              static_cast<size_t>(C * R) * sizeof(T)> {};

}  // namespace detail

template <class T>
inline constexpr bool is_direct_structured_data_v =
    detail::IsDirectStructuredData<std::remove_cv_t<T>>::value;

template <class T>
  requires is_direct_structured_data_v<T>
std::span<const std::byte> asStructuredBytes(std::span<const T> values) {
  return std::as_bytes(values);
}

template <class T>
  requires is_direct_structured_data_v<T>
std::span<std::byte> asWritableStructuredBytes(std::span<T> values) {
  return std::as_writable_bytes(values);
}

static_assert(is_direct_structured_data_v<glm::vec3>);
static_assert(is_direct_structured_data_v<glm::dvec3>);
static_assert(is_direct_structured_data_v<glm::mat3>);
static_assert(is_direct_structured_data_v<glm::dmat3>);
static_assert(is_direct_structured_data_v<glm::vec4>);
static_assert(is_direct_structured_data_v<glm::dvec4>);

}  // namespace sim::rhi
