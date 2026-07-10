//
// Created by creeper on 23-8-13.
//

#pragma once
#include <Core/core.h>
namespace ksk::core {
constexpr Real PI = 3.141592653589793238462643383279502884197169399375105820974;
constexpr Real PI_2 =
    1.570796326794896619231321691639751442098584699687552910487;
constexpr Real PI_4 =
    0.785398163397448309615660845819875721049292349843776455243;
constexpr Real PI_180 =
    0.017453292519943295769236907684886127134428718885417254560;
constexpr Real PI_180_INV =
    57.29577951308232087679815481410517033240547246656432154916;

using glm::normalize;
using glm::dot;

template<typename T>
#ifndef __CUDA_ARCH__
T clamp(const T &x, const T &min, const T &max) {
  return glm::clamp(x, min, max);
#else
CUDA_CALLABLE CUDA_FORCEINLINE T clamp(const T &x, const T &min, const T &max) {
  return fmaxf(min, fminf(x, max));
#endif
}

template<typename T>
inline T cubic(T x) { return x * x * x; }

template<typename T>
inline T sqr(T x) { return x * x; }

template<typename T>
inline T nearest(const T &A, const T &B, Real t) {
  assert(t >= 0.0 && t <= 1.0);
  return t <= 0.5 ? A : B;
}

template<typename T>
inline T lerp(const T &A, const T &B, Real t) {
  assert(t >= 0.0 && t <= 1.0);
  return (1.0 - t) * A + t * B;
}

template<typename T>
inline T catmullRomSpline(const T &A, const T &B, const T &C, const T &D,
                          Real t) {
  T d1 = (C - A) * 0.5;
  T d2 = (D - B) * 0.5;
  T D1 = C - B;
  T a0 = B;
  T a1 = d1;
  T a2 = 3 * D1 - 2 * d1 - d2;
  T a3 = d1 + d2 - 2 * D1;
  return a3 * cubic(t) + a2 * sqr(t) + a1 * t + a0;
}

template<typename T>
inline T bilerp(const T &AA, const T &BA, const T &AB, const T &BB,
                const Vector<T, 2> &t) {
  assert(t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0);
  return lerp(lerp(AA, BA, t.x), lerp(AB, BB, t.x), t.y);
}

template<typename T>
inline T trilerp(const T &AAA, const T &BAA, const T &ABA, const T &BBA,
                 const T &AAB, const T &BAB, const T &ABB, const T &BBB,
                 const Vector<T, 3> &t) {
  assert(t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0 && t.z >= 0.0 &&
      t.z <= 1.0);
  return lerp(bilerp(AAA, BAA, ABA, BBA, Vector<T, 2>(t.x, t.y)),
              bilerp(AAB, BAB, ABB, BBB, Vector<T, 2>(t.x, t.y)), t.z);
}

template<typename T>
inline T biCatmullRomSpline(const T &AA, const T &BA, const T &AB, const T &BB,
                            const Vector<T, 2> &t) {
  assert(t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0);
  return catmullRomSpline(catmullRomSpline(AA, BA, AB, BB, t.x),
                          catmullRomSpline(AA, BA, AB, BB, t.x), t.y);
}

template<typename T>
inline T triCatmullRomSpline(const T &AAA, const T &BAA, const T &ABA,
                             const T &BBA, const T &AAB, const T &BAB,
                             const T &ABB, const T &BBB,
                             const Vector<T, 3> &t) {
  assert(t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0 && t.z >= 0.0 &&
      t.z <= 1.0);
  return catmullRomSpline(
      biCatmullRomSpline(AAA, BAA, ABA, BBA, Vector<T, 2>(t.x, t.y)),
      biCatmullRomSpline(AAB, BAB, ABB, BBB, Vector<T, 2>(t.x, t.y)), t.z);
}

template<typename T, int Dim>
Vector<T, Dim> cwiseMin(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::min(a[i], b[i]);
  return result;
}

template<typename T, int Dim>
Vector<T, Dim> cwiseMax(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::max(a[i], b[i]);
  return result;
}

} // namespace core
// namespace core
