//
// Created by creeper on 5/31/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_MORTONS_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_MORTONS_H_
#include <cstdint>
#include <tuple>
#include <cassert>
namespace spatify {
inline uint32_t morton10BitEncode(uint32_t x) {
  x &= 0x3FF;
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

inline uint32_t morton10BitDecode(uint32_t x) {
  x = x & 0x09249249;
  x = (x | (x >> 2)) & 0x030C30C3;
  x = (x | (x >> 4)) & 0x0300F00F;
  x = (x | (x >> 8)) & 0x030000FF;
  x = (x | (x >> 16)) & 0x000003FF;
  return x;
}

inline std::tuple<uint32_t, uint32_t, uint32_t> decodeMorton10bit(uint32_t code) {
  uint32_t z = morton10BitDecode(code);
  uint32_t y = morton10BitDecode(code >> 1);
  uint32_t x = morton10BitDecode(code >> 2);
  return {x, y, z};
}

inline uint32_t encodeMorton10bit(int x, int y, int z) {
  assert(x >= 0 && y >= 0 && z >= 0);
  return (morton10BitEncode(x) << 2) | (morton10BitEncode(y) << 1) | morton10BitEncode(z);
}

inline uint64_t morton21BitEncode(uint32_t x) {
  uint64_t v = x;
  v &= 0x1fffff;
  v = (v | (v << 32)) & 0x7fff00000000ffffull;
  v = (v | (v << 16)) & 0x00ff0000ff0000ffull;
  v = (v | (v <<  8)) & 0x700f00f00f00f00full;
  v = (v | (v <<  4)) & 0x30c30c30c30c30c3ull;
  v = (v | (v <<  2)) & 0x1249249249249249ull;
  return v;
}
inline uint32_t morton21BitDecode(uint64_t x) {
  x = x & 0x1249249249249249ull;
  x = (x | (x >> 2)) & 0x30c30c30c30c30c3ull;
  x = (x | (x >> 4)) & 0x700f00f00f00f00full;
  x = (x | (x >> 8)) & 0x00ff0000ff0000ffull;
  x = (x | (x >> 16)) & 0x7fff00000000ffffull;
  x = (x | (x >> 32)) & 0x00000000001fffffull;
  return x;
}
inline std::tuple<uint32_t, uint32_t, uint32_t> decodeMorton21bit(uint64_t code) {
  uint32_t z = morton21BitDecode(code);
  uint32_t y = morton21BitDecode(code >> 1);
  uint32_t x = morton21BitDecode(code >> 2);
  return {x, y, z};
}
inline uint64_t encodeMorton21bit(int x, int y, int z) {
  assert(x >= 0 && y >= 0 && z >= 0);
  return (morton21BitEncode(x) << 2) | (morton21BitEncode(y) << 1) | morton21BitEncode(z);
}
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_MORTONS_H_
