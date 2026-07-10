//
// Created by creeper on 6/16/24.
//

#ifndef KISEKI_SPATIFY_INCLUDE_SPATIFY_PLATFORM_H_
#define KISEKI_SPATIFY_INCLUDE_SPATIFY_PLATFORM_H_

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
// GCC or Clang
#define HAS_BUILTIN_CLZ 1
#define HAS_BUILTIN_POPCOUNT 1
#elif defined(_MSC_VER)
// MSVC
    #include <intrin.h>
    #pragma intrinsic(__lzcnt)
    #pragma intrinsic(__popcnt)
    #define HAS_BUILTIN_CLZ 1
    #define HAS_BUILTIN_POPCOUNT 1
#else
    #define HAS_BUILTIN_CLZ 0
    #define HAS_BUILTIN_POPCOUNT 0
#endif

namespace spatify {
#if HAS_BUILTIN_CLZ
inline int countLeadingZeros32Bit(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return x ? __builtin_clz(x) : 32;
#elif defined(_MSC_VER)
  return x ? __lzcnt(x) : 32;
#endif
}

inline int countLeadingZeros64Bit(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return x ? __builtin_clzll(x) : 64;
#elif defined(_MSC_VER)
  return x ? __lzcnt64(x) : 64;
#endif
}
#else
// Fallback implementation
    inline int countLeadingZeros(uint32_t x) {
        if (x == 0) return 32;
        int n = 0;
        if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
        if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
        if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
        if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
        if (x <= 0x7FFFFFFF) { n += 1; }
        return n;
    }

    inline int countLeadingZeros(uint64_t x) {
        if (x == 0) return 64;
        int n = 0;
        if (x <= 0x00000000FFFFFFFF) { n += 32; x <<= 32; }
        if (x <= 0x0000FFFFFFFFFFFF) { n += 16; x <<= 16; }
        if (x <= 0x00FFFFFFFFFFFFFF) { n += 8; x <<= 8; }
        if (x <= 0x0FFFFFFFFFFFFFFF) { n += 4; x <<= 4; }
        if (x <= 0x3FFFFFFFFFFFFFFF) { n += 2; x <<= 2; }
        if (x <= 0x7FFFFFFFFFFFFFFF) { n += 1; }
        return n;
    }
#endif

#if HAS_BUILTIN_POPCOUNT
inline int popcount32Bit(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(x);
#elif defined(_MSC_VER)
  return __popcnt(x);
#endif
}

inline int popcount64Bit(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(x);
#elif defined(_MSC_VER)
  return __popcnt64(x);
#endif
}
#else
// Fallback implementation
    inline int popcount(uint32_t x) {
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        x = x + (x >> 8);
        x = x + (x >> 16);
        return x & 0x0000003F;
    }

    inline int popcount(uint64_t x) {
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        x = x + (x >> 8);
        x = x + (x >> 16);
        x = x + (x >> 32);
        return x & 0x0000007F;
    }
#endif
}
#endif //KISEKI_SPATIFY_INCLUDE_SPATIFY_PLATFORM_H_
