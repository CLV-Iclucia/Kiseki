//
// Created by creeper on 6/13/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASH_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASH_H_
#include <Spatify/spatial-hashable.h>
#include <Spatify/arrays.h>
#include <Spatify/mortons.h>
#include <iostream>
namespace spatify {

struct NeighbouringSphere {
  Real radius;
  Real x, y, z;
};

template<typename T>
class SpatialHash {
 public:
  using CoordType = T;

  template<SpatialHashablePrimitiveAccessor Accessor>
  void build(const Accessor &primitives, Real spacing) {
    h = spacing;
    auto num_primitives = primitives.size();
    BBox<CoordType, 3> scene_bound{};
    for (int i = 0; i < num_primitives; i++)
      scene_bound = scene_bound.merge(primitives.bbox(i));
    m_origin = scene_bound.lo;
    int x_res = static_cast<int>(std::ceil(scene_bound.extent().x / h));
    int y_res = static_cast<int>(std::ceil(scene_bound.extent().y / h));
    int z_res = static_cast<int>(std::ceil(scene_bound.extent().z / h));
    m_ranges.resize(x_res * y_res * z_res);
    m_resolution = {x_res, y_res, z_res};
    std::vector<std::pair<uint64_t, int>> grid_id_pairs{};
    for (int i = 0; i < num_primitives; i++) {
      auto bbox = primitives.bbox(i);
      auto ox = static_cast<int>(std::floor(bbox.lo.x / h));
      auto oy = static_cast<int>(std::floor(bbox.lo.y / h));
      auto oz = static_cast<int>(std::floor(bbox.lo.z / h));
      for (auto it = primitives.pruningIterator(i, {0.0, 0.0, 0.0, h}); !it.end(); ++it) {
        auto x = it.xOffset() + ox;
        auto y = it.yOffset() + oy;
        auto z = it.zOffset() + oz;
        if (x > x_res || y > y_res || z > z_res) continue;
        uint64_t code = encodeMorton21bit(x, y, z);
        grid_id_pairs.push_back({code, i});
      }
    }
    std::ranges::sort(grid_id_pairs.begin(), grid_id_pairs.end());
    Range range{-1, -1};
    std::fill(m_ranges.begin(), m_ranges.end(), range);
    m_primitive_indices.resize(grid_id_pairs.size());
    for (int i = 0; i < grid_id_pairs.size(); i++) {
      const auto &[code, idx] = grid_id_pairs[i];
      m_primitive_indices[i] = idx;
      auto [x, y, z] = decodeMorton21bit(code);
      if (i == 0) range.begin = i;
      if (i == grid_id_pairs.size() - 1 || code != grid_id_pairs[i + 1].first) {
        range.end = i + 1;
        m_ranges[index(x, y, z)] = range;
        range.begin = i + 1;
      }
    }
  }
  template<typename Func>
  void forNeighbouringPrimitives(const NeighbouringSphere &sphere, Func &&func) {
    auto [radius, x, y, z] = sphere;
    int x_min = static_cast<int>(std::floor((x - radius) / h));
    int x_max = static_cast<int>(std::ceil((x + radius) / h));
    int y_min = static_cast<int>(std::floor((y - radius) / h));
    int y_max = static_cast<int>(std::ceil((y + radius) / h));
    int z_min = static_cast<int>(std::floor((z - radius) / h));
    int z_max = static_cast<int>(std::ceil((z + radius) / h));
    for (int i = x_min; i <= x_max; i++) {
      for (int j = y_min; j <= y_max; j++) {
        for (int k = z_min; k <= z_max; k++) {
          auto [begin, end] = m_ranges[index(i, j, k)];
          for (int idx = begin; idx < end; idx++)
            func(m_primitive_indices[idx]);
        }
      }
    }
  }
  template<typename Func>
  void forNeighbouringPrimitives(const BBox<Real, 3>& querying_box, Func &&func) {
    auto x_min = static_cast<int>(std::floor(querying_box.lo.x / h));
    auto x_max = static_cast<int>(std::ceil(querying_box.hi.x / h));
    auto y_min = static_cast<int>(std::floor(querying_box.lo.y / h));
    auto y_max = static_cast<int>(std::ceil(querying_box.hi.y / h));
    auto z_min = static_cast<int>(std::floor(querying_box.lo.z / h));
    auto z_max = static_cast<int>(std::ceil(querying_box.hi.z / h));
    for (int i = x_min; i <= x_max; i++) {
      for (int j = y_min; j <= y_max; j++) {
        for (int k = z_min; k <= z_max; k++) {
          auto [begin, end] = m_ranges[index(i, j, k)];
          for (int idx = begin; idx < end; idx++)
            func(m_primitive_indices[idx]);
        }
      }
    }
  }
  template<typename Func>
  void forProbablePrimitivesInCell(int i, int j, int k, Func &&func) {
    auto [begin, end] = m_ranges[index(i, j, k)];
    for (int idx = begin; idx < end; idx++)
      func(m_primitive_indices[idx]);
  }
  [[nodiscard]] Vec3i resolution() const {
    return m_resolution;
  }
 private:
  struct Range {
    int begin;
    int end;
  };
  [[nodiscard]] int index(int x, int y, int z) const {
    assert(x < m_resolution.x && y < m_resolution.y && z < m_resolution.z);
    assert(x >= 0 && y >= 0 && z >= 0);
    return x + y * m_resolution.x + z * m_resolution.x * m_resolution.y;
  }
  CoordType h{};
  Vector<CoordType, 3> m_origin{};
  Vec3i m_resolution{};
  std::vector<Range> m_ranges{};
  std::vector<int> m_primitive_indices{};
};

}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_SPATIAL_HASH_H_
