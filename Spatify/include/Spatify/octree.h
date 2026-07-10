//
// Created by creeper on 6/16/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_OCTREE_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_OCTREE_H_
#include <array>
#include <memory>
#include <optional>
#include <Spatify/bbox.h>
#include <Spatify/platform.h>
namespace spatify {

template<typename T>
class Octree {
  struct Node {
    const Node *parent{};
    std::optional<std::array<std::unique_ptr<Node>, 8>> children{};
    uint8_t validChildren{};
    BBox<T, 3> bbox{};
    [[nodiscard]] bool isLeaf() const {
      return children.has_value();
    }
    [[nodiscard]] int countChildren() const {
      return popcount32Bit(validChildren);
    }
    void subdivide() {
      children = std::make_optional<std::array<std::unique_ptr<Node>, 8>>();
      auto& children_array = *children;
      for (int i = 0; i < 8; ++i) {
        (*children)[i] = std::make_unique<Node>();
        (*children)[i]->parent = this;
      }

    }
  };
  std::unique_ptr<Node> root;
};
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_OCTREE_H_
