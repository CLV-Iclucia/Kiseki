//
// Created by creeper on 5/23/24.
//
#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_QUERY_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_QUERY_H_
#include <Spatify/bbox.h>
namespace spatify {
template <typename T>
concept SpatialQuery = requires(T query, int id) {
  { query(id) } -> std::same_as<bool>;
};

template <typename T>
concept BBoxIntersectionQuery = requires(T query) {
  { query(BBox<typename T::CoordType, 3>()) } -> std::same_as<bool>;
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_QUERY_H_
