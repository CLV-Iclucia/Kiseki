//
// Created by creeper on 5/30/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPATIAL_HASH_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPATIAL_HASH_H_
#include <Spatify/sparse/sparse-array.h>
#include <Spatify/parallel.h>
#include <Spatify/spatial-hashable.h>
namespace spatify {
// SpatialHash is a sparse data structure that can be used for spatial query acceleration.
// For performance reasons, we suggest that the accessor be small
template <SpatialHashablePrimitive Primitive>
class SpatialHash {
 public:
  using CoordType = typename Primitive::CoordType;
  template <SpatialHashablePrimitiveAccessor Accessor>
  requires std::convertible_to<typename Accessor::PrimitiveType, Primitive>
  void build(const Accessor &accessor, Real spacing) {
    m_primitive_indices.resize(accessor.size());
    h = spacing;
    auto num_primitives = accessor.size();
    std::vector<std::pair<uint64_t, int>> m_grid_id_pair{};
    for (int i = 0; i < num_primitives; i++) {
      auto bbox = accessor.primitive(i).bbox();
      auto ox = static_cast<int>(bbox.lo.x / h) * h;
      auto oy = static_cast<int>(bbox.lo.y / h) * h;
      auto oz = static_cast<int>(bbox.lo.z / h) * h;
      for (auto it = accessor.primitive(i).rasterize({h, ox, oy, oz}); !it.end(); ++it) {
        uint64_t code = encodeMorton21bit(it.x, it.y, it.z);
        m_grid_id_pair.push_back({code, i});
      }
    }
    std::sort(m_grid_id_pair.begin(), m_grid_id_pair.end());
    Range range{-1, -1};
    for (int i = 0; i < m_grid_id_pair.size(); i++) {
      const auto& [code, idx] = m_grid_id_pair[i];
      m_primitive_indices[i] = idx;
      auto [x, y, z] = decodeMorton21bit(code);
      if (i == 0) range.begin = i;
      if (i == m_grid_id_pair.size() - 1 || code != m_grid_id_pair[i + 1].first) {
        range.end = i + 1;
        m_range.write(x, y, z, range);
        range.begin = i + 1;
      }
    }
  }
  template <typename Func>
  void forNeighbouringPrimitives(Real x, Real y, Real z, Real radius, Func&& func) {
    int x_min = static_cast<int>(std::floor((x - radius) / h));
    int x_max = static_cast<int>(std::ceil((x + radius) / h));
    int y_min = static_cast<int>(std::floor((y - radius) / h));
    int y_max = static_cast<int>(std::ceil((y + radius) / h));
    int z_min = static_cast<int>(std::floor((z - radius) / h));
    int z_max = static_cast<int>(std::ceil((z + radius) / h));
    for (int i = x_min; i <= x_max; i++) {
      for (int j = y_min; j <= y_max; j++) {
        for (int k = z_min; k <= z_max; k++) {
          auto opt_range = m_range.tryRead(i, j, k);
          if (!opt_range.has_value()) continue;
          const auto& [begin, end] = *opt_range;
          for (int idx = begin; idx <= end; idx++)
            func(m_primitive_indices[idx]);
        }
      }
    }
  }
 private:
  struct Range {
    int begin;
    int end;
  };
  CoordType h{};
  SparseArray<Range> m_range{};
  std::vector<int> m_primitive_indices{};
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPATIAL_HASH_H_
