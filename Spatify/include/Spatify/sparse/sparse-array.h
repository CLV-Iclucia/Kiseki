//
// Created by creeper on 5/28/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPARSE_ARRAY_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPARSE_ARRAY_H_
#include <Spatify/types.h>
#include <Spatify/mortons.h>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <optional>
namespace spatify {
inline int popCount(uint64_t x) {
#if(defined(__GNUC__) || defined(__clang__))
  return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}
template<int logN, typename Node>
struct PointerNode {
  using DataType = Node::DataType;
  static constexpr uint32_t Resolution = Node::Resolution << logN;
  std::array<std::unique_ptr<Node>, (1 << (3 * logN))> chunk{};
  static constexpr int BlockSize = (Node::BlockSize) << logN;
  void clear() {
    memset(chunk.data(), 0, sizeof(chunk));
  }
  std::optional<DataType> tryRead(int x, int y, int z) {
    int bx = x >> logN;
    int by = y >> logN;
    int bz = z >> logN;
    int x_offset = x & ((1 << logN) - 1);
    int y_offset = y & ((1 << logN) - 1);
    int z_offset = z & ((1 << logN) - 1);
    auto index = bx << (logN << 1) | (by << logN) | bz;
    if (bx >= (1 << logN) || by >= (1 << logN) || bz >= (1 << logN) || chunk[index] == nullptr)
      return std::nullopt;
    return chunk[index]->tryRead(x_offset, y_offset, z_offset);
  }
  DataType read(int x, int y, int z) {
    int bx = x >> logN;
    int by = y >> logN;
    int bz = z >> logN;
    int x_offset = x & ((1 << logN) - 1);
    int y_offset = y & ((1 << logN) - 1);
    int z_offset = z & ((1 << logN) - 1);
    auto index = bx << (logN << 1) | (by << logN) | bz;
    assert(bx < (1 << logN) && by < (1 << logN) && bz < (1 << logN) && chunk[index] != nullptr);
    return chunk[index]->read(x_offset, y_offset, z_offset);
  }
  void touch(int bx, int by, int bz) {
    auto index = bx << (logN << 1) | (by << logN) | bz;
    if (chunk[index] == nullptr) {
      chunk[index] = std::make_unique<Node>();
      chunk[index]->clear();
    }
  }
  void write(int x, int y, int z, const DataType &value) {
    int bx = x >> logN;
    int by = y >> logN;
    int bz = z >> logN;
    int x_offset = x & ((1 << logN) - 1);
    int y_offset = y & ((1 << logN) - 1);
    int z_offset = z & ((1 << logN) - 1);
    auto index = bx << (logN << 1) | (by << logN) | bz;
    assert(bx < (1 << logN) && by < (1 << logN) && bz < (1 << logN));
    touch(bx, by, bz);
    chunk[index]->write(x_offset, y_offset, z_offset, value);
  }
  int validCount() {
    int count = 0;
    for (auto &node : chunk)
      if (node != nullptr)
        count += node->validCount();
    return count;
  }
  int validChildren() {
    int count = 0;
    for (auto &node : chunk)
      if (node != nullptr)
        count++;
    return count;
  }
  [[nodiscard]] bool valid(int x, int y, int z) const {
    int bx = x >> logN;
    int by = y >> logN;
    int bz = z >> logN;
    int x_offset = x & ((1 << logN) - 1);
    int y_offset = y & ((1 << logN) - 1);
    int z_offset = z & ((1 << logN) - 1);
    auto index = bx << (logN << 1) | (by << logN) | bz;
    return bx < (1 << logN) && by < (1 << logN) && bz < (1 << logN) && chunk[index] != nullptr
        && chunk[index]->valid(x_offset, y_offset, z_offset);
  }
  [[nodiscard]] bool validChild(int bx, int by, int bz) const {
    auto index = bx << (logN << 1) | (by << logN) | bz;
    return chunk[index] != nullptr;
  }
};

template<int N, typename T>
struct DenseNode {
  using DataType = T;
  std::array<T, N * N * N> chunk;
  uint64_t valid_bit_mask[(N * N * N) >> 6]{0};
  static constexpr int BlockSize = N;
  static constexpr uint32_t Resolution = N;
  bool valid(int x, int y, int z) {
    auto index = encodeMorton10bit(x, y, z);
    assert(index < N * N * N);
    return valid_bit_mask[index >> 6] & (1 << (index & 63));
  }
  std::optional<DataType> tryRead(int x, int y, int z) {
    auto index = encodeMorton10bit(x, y, z);
    assert(index < N * N * N);
    if (!(valid_bit_mask[index >> 6] & (1 << (index & 63))))
      return std::nullopt;
    return chunk[index];
  }
  T read(int x, int y, int z) {
    auto index = encodeMorton10bit(x, y, z);
    assert(index < N * N * N);
    return chunk[index];
  }
  void write(int x, int y, int z, const T &value) {
    auto index = encodeMorton10bit(x, y, z);
    assert(index < N * N * N);
    chunk[index] = value;
    valid_bit_mask[index >> 6] |= (1 << (index & 63));
  }
  int validCount() {
    int count = 0;
    for (auto mask : valid_bit_mask)
      count += popCount(mask);
    return count;
  }
  void clear() {
    memset(valid_bit_mask, 0, sizeof(valid_bit_mask));
    memset(chunk.data(), 0, sizeof(chunk));
  }
};

template<typename Node>
struct HashNode {
  // create self-defined hash for unordered_map
  struct KeyHash {
    std::size_t operator()(const std::tuple<int, int, int> &key) const {
      const auto& [x, y, z] = key;
      return (x * 11) ^ (y * 45) ^ (z * 14);
    }
  };
  std::unordered_map<std::tuple<int, int, int>, std::unique_ptr<Node>, KeyHash> map;
  static constexpr uint32_t Resolution = Node::Resolution;
  using DataType = Node::DataType;
  std::optional<DataType> tryRead(int x, int y, int z) {
    auto key = std::make_tuple(x, y, z);
    if (!valid(x, y, z))
      return std::nullopt;
    return map[key]->tryRead(x, y, z);
  }
  DataType read(int x, int y, int z) {
    auto key = std::make_tuple(x, y, z);
    assert(map.contains(key));
    return map[key]->read(x, y, z);
  }
  void write(int x, int y, int z, const DataType &value) {
    auto key = std::make_tuple(x, y, z);
    if (!map.contains(key)) {
      map[key] = std::make_unique<Node>();
      map[key]->clear();
    }
    map[key]->write(x, y, z, value);
  }
  int validCount() {
    int count = 0;
    for (auto &node : map)
      count += node.second->validCount();
    return count;
  }
  [[nodiscard]] bool valid(int x, int y, int z) const {
    auto key = std::make_tuple(x, y, z);
    return map.contains(key) && map.at(key)->valid(x, y, z);
  }
};

template<typename T>
using DefaultStructure = HashNode<PointerNode<4, DenseNode<4, T>>>;

template<typename T, typename NodeStructure = DefaultStructure<T>>
struct SparseArray {
  std::unique_ptr<NodeStructure> root{};
  using DataType = T;
  static constexpr uint32_t Resolution = NodeStructure::Resolution;
  std::optional<DataType> tryRead(int x, int y, int z) {
    return root->tryRead(x, y, z);
  }
  DataType read(int x, int y, int z) {
    return root->read(x, y, z);
  }
  void write(int x, int y, int z, const DataType &value) {
    root->write(x, y, z, value);
  }
  [[nodiscard]] bool valid(int x, int y, int z) const {
    return root->valid(x, y, z);
  }
  int validCount() {
    return root->validCount();
  }
};

}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_SPARSE_SPARSE_ARRAY_H_
