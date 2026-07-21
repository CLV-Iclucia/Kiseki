#ifndef SPATIFY_INCLUDE_SPATIFY_BBOX_H
#define SPATIFY_INCLUDE_SPATIFY_BBOX_H
#include <Spatify/types.h>
namespace spatify {

template <typename T, int Dim>
Vector<T, Dim> cwiseMin(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::min(a[i], b[i]);
  return result;
}

template <typename T, int Dim>
Vector<T, Dim> cwiseMax(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::max(a[i], b[i]);
  return result;
}

template<typename T, int Dim>
struct BBox {
  Vector<T, Dim> lo{std::numeric_limits<T>::max()};
  Vector<T, Dim> hi{std::numeric_limits<T>::lowest()};
  BBox() = default;
  explicit BBox(const Vector<T, Dim> &p) : lo(p), hi(p) {}
  BBox(const Vector<T, Dim> &lo, const Vector<T, Dim> &hi) : lo(lo), hi(hi) {
  }
  BBox& operator=(const BBox &other) = default;
  BBox merge(const BBox &other) const {
    return {cwiseMin<T, Dim>(lo, other.lo), cwiseMax<T, Dim>(hi, other.hi)};
  }
  BBox &expand(const Vector<T, 3> &p) {
    lo = cwiseMin<T, Dim>(lo, p);
    hi = cwiseMax<T, Dim>(hi, p);
    return *this;
  }
  BBox &expand(const BBox &other) {
    lo = cwiseMin<T, Dim>(lo, other.lo);
    hi = cwiseMax<T, Dim>(hi, other.hi);
    return *this;
  }
  BBox& dilate(T factor) {
    lo -= factor;
    hi += factor;
    return *this;
  }
  BBox dilate(T factor) const {
    return {lo - factor, hi + factor};
  }
  bool overlap(const BBox &other) const {
    bool overlap_x = lo.x <= other.hi.x && hi.x >= other.lo.x;
    bool overlap_y = lo.y <= other.hi.y && hi.y >= other.lo.y;
    bool overlap_z = lo.z <= other.hi.z && hi.z >= other.lo.z;
    return overlap_x && overlap_y && overlap_z;
  }
  bool contains(const Vector<T, Dim> &point) const {
    return (point.array() >= lo.array()).all() &&
        (point.array() <= hi.array()).all();
  }
  Vector<T, Dim> centre() const {
    return (lo + hi) * static_cast<T>(0.5);
  }
  Vector<T, Dim> extent() const {
    return hi - lo;
  }
};
}
#endif //SPATIFY_INCLUDE_SPATIFY_BBOX_H
