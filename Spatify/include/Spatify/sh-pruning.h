//
// Created by creeper on 5/31/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_RASTERIZERS_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_RASTERIZERS_H_
// some frequently used rasterizers for point, edge and triangle
#include <Spatify/bbox.h>
namespace spatify {

template<typename T>
struct PointIterator {
  int x, y, z;
  bool has_increased{false};
  PointIterator(int x, int y, int z) : x(x), y(y), z(z) {}
  [[nodiscard]] bool end() const { return has_increased; }
  void operator++() { has_increased = true; }
};

template<typename T>
struct BBoxIterator {
  BBoxIterator(const BBox<T, 3>& bbox, const SingleCell<T>& cell) {
    auto extent = bbox.extent();
    auto [ox, oy, oz, h] = cell;
    x_extent = std::ceil((extent.x - ox) / h);
    y_extent = std::ceil((extent.y - oy) / h);
    z_extent = std::ceil((extent.z - oz) / h);
  }
  [[nodiscard]] bool end() const {
    return ended;
  }
  BBoxIterator& operator++() {
    if (++x == x_extent) {
      x = 0;
      if (++y == y_extent) {
        y = 0;
        if (++z == z_extent)
          ended = true;
      }
    }
    return *this;
  }
  [[nodiscard]] int xOffset() const {
    return x;
  }
  [[nodiscard]] int yOffset() const {
    return y;
  }
  [[nodiscard]] int zOffset() const {
    return z;
  }
 private:
  int x{}, y{}, z{};
  int x_extent{}, y_extent{}, z_extent{};
  bool ended{false};
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_RASTERIZERS_H_
