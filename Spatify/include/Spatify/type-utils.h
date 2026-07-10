//
// Created by creeper on 23-8-15.
//

#ifndef KISEKI_CORE_INCLUDE_CORE_TYPE_UTILS_H_
#define KISEKI_CORE_INCLUDE_CORE_TYPE_UTILS_H_
#include <type_traits>
namespace spatify {
template <int a, int b> struct compile_time_gcd {
  static_assert(a >= 0 && b >= 0);
  static constexpr int value = compile_time_gcd<b, a % b>::value;
};

template <int a> struct compile_time_gcd<a, 0> {
  static_assert(a >= 0);
  static constexpr int value = a;
};

// this enables us to use fractions in template arguments!
template <int Numerator, int Denominator> struct Fraction {
  static_assert(Denominator != 0, "Denominator cannot be zero");
  static_assert(compile_time_gcd<Numerator, Denominator>::value == 1 || Numerator == 0,
                "Fraction must be irreducible");
  static constexpr Real value = static_cast<Real>(Numerator) / Denominator;
};

// TODO: if needed, add support for vectors of higher dimensions
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY>
struct compile_time_vec2 {
  static constexpr Real x = Fraction<NumeratorX, DenominatorX>::value;
  static constexpr Real y = Fraction<NumeratorY, DenominatorY>::value;
};

template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY,
          int NumeratorZ, int DenominatorZ>
struct compile_time_vec3 {
  static constexpr Real x = Fraction<NumeratorX, DenominatorX>::value;
  static constexpr Real y = Fraction<NumeratorY, DenominatorY>::value;
  static constexpr Real z = Fraction<NumeratorZ, DenominatorZ>::value;
};

template <typename T, int Dim> struct is_compile_time_vec {
  static constexpr bool value = false;
};
template <int Dim>
using Zeros = std::conditional_t<Dim == 2, compile_time_vec2<0, 1, 0, 1>,
                                 compile_time_vec3<0, 1, 0, 1, 0, 1>>;
template <int Dim>
using Halfs = std::conditional_t<Dim == 2, compile_time_vec2<1, 2, 1, 2>,
                                 compile_time_vec3<1, 2, 1, 2, 1, 2>>;
template <int Dim, int Axis>
using Half = std::conditional_t<Dim == 2, compile_time_vec2<Axis == 0, 2,
                                                           Axis == 1, 2>,
                                compile_time_vec3<Axis == 0, 2, Axis == 1, 2,
                                                  Axis == 2, 2>>;

// specialize is_compile_time_vec for compile_time_vec2 and compile_time_vec3
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY>
struct is_compile_time_vec<
    compile_time_vec2<NumeratorX, DenominatorX, NumeratorY, DenominatorY>, 2> {
  static constexpr bool value = true;
};
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY,
          int NumeratorZ, int DenominatorZ>
struct is_compile_time_vec<
    compile_time_vec3<NumeratorX, DenominatorX, NumeratorY, DenominatorY,
                      NumeratorZ, DenominatorZ>,
    3> {
  static constexpr bool value = true;
};

} // namespace core
#endif // KISEKI_CORE_INCLUDE_CORE_TYPE_UTILS_H_
