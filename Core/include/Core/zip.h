//
// Created by creeper on 6/3/24.
//

#pragma once
#include <concepts>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>
namespace ksk::core {
template<typename T>
concept Iterator = requires(T it) {
  requires std::is_convertible_v<std::decay_t<decltype(*it)>, typename T::value_type>;
  typename T::reference;
  { ++it } -> std::convertible_to<T>;
  { it != it } -> std::convertible_to<bool>;
  { it == it } -> std::convertible_to<bool>;
};

template<typename T>
concept Iterable = requires(T t) {
  { std::begin(t) } -> Iterator;
  { std::end(t) } -> Iterator;
};

template <typename T>
using select_access_t = std::conditional_t<
    std::is_same_v<T, std::vector<bool>::iterator> || std::is_same_v<T, std::vector<bool>::const_iterator>,
    typename T::value_type,
    typename T::reference
>;

template<Iterator... Iters>
class ZipIterator {
  std::tuple<Iters...> m_iters;
 public:
  using value_type = std::tuple<select_access_t<Iters>...>;
  ZipIterator() = delete;
  explicit ZipIterator(Iters &&... iters) : m_iters(std::forward<Iters>(iters)...) {}

  ZipIterator &operator++() {
    std::apply([](auto &... iters) { (++iters, ...); }, m_iters);
    return *this;
  }
  bool operator==(const ZipIterator &other) const {
    return m_iters == other.m_iters;
  }
  bool operator!=(const ZipIterator &other) const {
    return *this != other;
  }
  value_type operator*() {
    return std::apply([](auto &&... iters) { return value_type(*iters...); }, m_iters);
  }
};

template<typename T>
using select_iterator_t = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
                                             typename std::decay_t<T>::const_iterator,
                                             typename std::decay_t<T>::iterator>;
template<Iterable... Iterables>
struct ZipRange {
 public:
  using iterator_t = ZipIterator<select_iterator_t<Iterables>...>;
  std::tuple<Iterables ...> iterables;
  template<Iterable... Args>
  explicit ZipRange(Args &&... iterables) : iterables(std::forward<Args>(iterables)...) {}
  iterator_t begin() {
    return std::apply([](auto &&... args) {
      return (iterator_t(std::begin(args)...)); }, iterables);
  }
  iterator_t end() {
    return std::apply([](auto &&... args) {
      return (iterator_t(std::end(args)...)); }, iterables);
  }
};
template<Iterable... Iterables>
auto zip(Iterables &&... iterables) {
  return ZipRange<Iterables...>(std::forward<Iterables>(iterables)...);
}
}
