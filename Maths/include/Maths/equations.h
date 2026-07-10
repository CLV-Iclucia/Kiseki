//
// Created by creeper on 5/28/24.
//

#ifndef KISEKI_MATHS_INCLUDE_MATHS_EQUATIONS_H_
#define KISEKI_MATHS_INCLUDE_MATHS_EQUATIONS_H_

#include <optional>
#include <iostream>
#include <Maths/types.h>
namespace ksk::maths {
struct QuadraticPolynomial {
  Real a, b, c;
};

struct CubicEquationRoots {
  bool infiniteSolutions{};
  std::array<Real, 3> roots{};
  int numRoots{};
  CubicEquationRoots() = default;
  explicit CubicEquationRoots(bool infinite) : infiniteSolutions(infinite) {}
  explicit CubicEquationRoots(Real x) : numRoots(1) {
    roots[0] = x;
  }
  void addRoot(Real x) {
    roots[numRoots++] = x;
  }
};


CubicEquationRoots quadraticSolve(const QuadraticPolynomial &poly);

struct CubicPolynomial {
  Real a, b, c, d;
};

inline CubicEquationRoots infiniteCubicRoots() {
  return CubicEquationRoots{true};
}

struct BinaryLinearSolution {
  Real x{}, y{};
};

CubicEquationRoots clampedCubicSolve(const CubicPolynomial &poly, Real l, Real r, Real tolerance);

struct BinaryLinearSystem {
  Real a00, a01, a10, a11, b0, b1;
};

inline std::optional<BinaryLinearSolution> binaryLinearSolve(const BinaryLinearSystem &sys) {
  Real det = sys.a00 * sys.a11 - sys.a01 * sys.a10;
  if (det == 0.0)
    return std::nullopt;
  Real x = (sys.b0 * sys.a11 - sys.b1 * sys.a01) / det;
  Real y = (sys.a00 * sys.b1 - sys.a10 * sys.b0) / det;
  return std::make_optional<BinaryLinearSolution>({x, y});
}

inline bool hasInfiniteSolutions(const BinaryLinearSystem &sys) {
  return sys.a00 * sys.a11 == sys.a01 * sys.a10 && sys.a00 * sys.b1 == sys.a10 * sys.b0;
}
}
#endif //KISEKI_MATHS_INCLUDE_MATHS_EQUATIONS_H_
