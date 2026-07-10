//
// cross-lang.h — C++/HLSL cross-language compatibility shim.
//
// Include this first in any header whose computational core must be shared
// verbatim between CPU code and compute shaders. `__cplusplus` is defined by
// C++ compilers and NOT by DXC, so the correct branch is selected
// automatically depending on who #includes the file.
//
// The shared subset:
//   sh_real      double scalar
//   sh_real3     3-vector of double         (glm::dvec3 / HLSL double3)
//   SH_INLINE    `inline`                   / nothing
//   SH_OUT(T)    `T&`                        / `out T`
//   SH_ASSERT(x) `assert(x)`                / nothing
//   SH_NS_BEGIN/END  wrap shared symbols in `ksk::fem::shared` on the CPU,
//                    expand to nothing in HLSL (which has no namespaces).
//
// Vector ops (shDot / shCross / shMax) are written as explicit scalar double
// expressions ON PURPOSE: HLSL's dot()/cross() intrinsics, like sqrt(), are
// float-only and silently downcast double arguments, which would break
// bit-for-bit agreement with the CPU. Keep everything in full double here.
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_CROSS_LANG_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_CROSS_LANG_H_

#ifdef __cplusplus
  // ----------------------------- C++ ------------------------------------
  #include <glm/glm.hpp>
  #include <cassert>
  #include <cmath>

  namespace ksk::fem::shared {
  using sh_real  = double;
  using sh_real3 = glm::dvec3;
  }  // namespace ksk::fem::shared

  #define SH_NS_BEGIN  namespace ksk::fem::shared {
  #define SH_NS_END    }
  #define SH_INLINE    inline
  #define SH_OUT(T)    T&
  #define SH_ASSERT(x) assert(x)
  #define SH_LOOP
  #define SH_UNROLL
#else
  // ----------------------------- HLSL (DXC) -----------------------------
  #include <RHI/structured-buffer-access.hlsli>
  #define sh_real      double
  #define sh_real3     double3
  #define SH_NS_BEGIN
  #define SH_NS_END
  #define SH_INLINE
  #define SH_OUT(T)    out T
  #define SH_ASSERT(x)
  #define SH_LOOP      [loop]
  #define SH_UNROLL    [unroll]
#endif

SH_NS_BEGIN

// Full-precision double dot product (do NOT use the float-only dot() intrinsic).
SH_INLINE sh_real shDot(sh_real3 a, sh_real3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Full-precision double cross product (do NOT use the float-only cross() intrinsic).
SH_INLINE sh_real3 shCross(sh_real3 a, sh_real3 b) {
    return sh_real3(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

SH_INLINE sh_real shMax(sh_real a, sh_real b) { return a > b ? a : b; }

// Full-precision double sqrt. HLSL's sqrt() intrinsic is float-only, so on the
// GPU we seed with the float sqrt and refine with two Newton iterations
// y <- 0.5*(y + x/y), recovering full double precision (same trick as dsqrt).
SH_INLINE sh_real shSqrt(sh_real x) {
#ifdef __cplusplus
    return std::sqrt(x);
#else
    if (x <= 0.0) return 0.0;
    sh_real y = (sh_real)sqrt((float)x);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    return y;
#endif
}

// Full-precision double natural log. HLSL's log() is float-only, so on the GPU
// we range-reduce x = m * 2^e (m in [1,2) via the IEEE-754 exponent bits, then
// folded to [sqrt(1/2), sqrt(2)) so |s|<=0.172) and evaluate the atanh series
//   ln(m) = 2*(s + s^3/3 + s^5/5 + ...),  s = (m-1)/(m+1)
// to 11 terms (full double for |s|<=0.172), then add e*ln2.
SH_INLINE sh_real shLog(sh_real x) {
#ifdef __cplusplus
    return std::log(x);
#else
    if (x <= 0.0) return -1e300;
    uint lo, hi;
    asuint(x, lo, hi);
    int e = (int)((hi >> 20) & 0x7FFu) - 1023;
    uint mhi = (hi & 0x000FFFFFu) | 0x3FF00000u;  // force exponent 0 -> m in [1,2)
    sh_real m = asdouble(lo, mhi);
    if (m > 1.4142135623730951) { m *= 0.5; e += 1; }
    sh_real s  = (m - 1.0) / (m + 1.0);
    sh_real s2 = s * s;
    sh_real p  = 2.0 / 21.0;
    p = p * s2 + 2.0 / 19.0;
    p = p * s2 + 2.0 / 17.0;
    p = p * s2 + 2.0 / 15.0;
    p = p * s2 + 2.0 / 13.0;
    p = p * s2 + 2.0 / 11.0;
    p = p * s2 + 2.0 / 9.0;
    p = p * s2 + 2.0 / 7.0;
    p = p * s2 + 2.0 / 5.0;
    p = p * s2 + 2.0 / 3.0;
    p = p * s2 + 2.0;
    return (sh_real)e * 0.69314718055994530942 + s * p;
#endif
}

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_CROSS_LANG_H_
