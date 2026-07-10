//
// Created by creeper on 6/13/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASHABLE_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASHABLE_H_
#include <Spatify/bbox.h>
namespace spatify {

template<typename T>
struct SingleCell {
  Real ox, oy, oz, h;
};

template<typename T>
concept SHBroadPhasePruningIterator = requires(T t) {
  { t.xOffset() } -> std::convertible_to<int>;
  { t.yOffset() } -> std::convertible_to<int>;
  { t.zOffset() } -> std::convertible_to<int>;
  { t.end() } -> std::convertible_to<bool>;
  { ++t } -> std::same_as<T &>;
};

// this accessor had better be trivially copyable
template<typename T>
concept SpatialHashablePrimitiveAccessor = requires(T t, size_t index, const SingleCell<typename T::CoordType> &cell) {
  { t.bbox(index) } -> std::convertible_to<BBox<typename T::CoordType, 3>>;
  requires SHBroadPhasePruningIterator<decltype(t.pruningIterator(index, cell))>;
  { t.size() } -> std::convertible_to<size_t>;
};

}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASHABLE_H_
