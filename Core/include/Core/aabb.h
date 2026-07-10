#pragma once
#include <Core/core.h>
#include <algorithm>
#include <limits>

namespace ksk::core {

template <typename T, int Dim>
struct AABB {
  Vector<T, Dim> lower{std::numeric_limits<T>::max()};
  Vector<T, Dim> upper{std::numeric_limits<T>::lowest()};

  void expand(const Vector<T, Dim> &point) {
    for (int i = 0; i < Dim; ++i) {
      lower[i] = std::min(lower[i], point[i]);
      upper[i] = std::max(upper[i], point[i]);
    }
  }

  void expand(const AABB &other) {
    for (int i = 0; i < Dim; ++i) {
      lower[i] = std::min(lower[i], other.lower[i]);
      upper[i] = std::max(upper[i], other.upper[i]);
    }
  }

  [[nodiscard]] bool contains(const Vector<T, Dim> &point) const {
    for (int i = 0; i < Dim; ++i)
      if (point[i] < lower[i] || point[i] > upper[i])
        return false;
    return true;
  }

  [[nodiscard]] bool overlaps(const AABB &other) const {
    for (int i = 0; i < Dim; ++i)
      if (lower[i] > other.upper[i] || upper[i] < other.lower[i])
        return false;
    return true;
  }

  [[nodiscard]] Vector<T, Dim> center() const {
    return (lower + upper) * T(0.5);
  }

  [[nodiscard]] Vector<T, Dim> extent() const {
    return upper - lower;
  }
};

using AABB3d = AABB<Real, 3>;
using AABB3f = AABB<float, 3>;
using AABB2d = AABB<Real, 2>;

} // namespace ksk::core
