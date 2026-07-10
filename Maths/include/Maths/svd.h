//
// Created by creeper on 5/25/24.
//

#ifndef KISEKI_MATHS_INCLUDE_MATHS_SVD_H_
#define KISEKI_MATHS_INCLUDE_MATHS_SVD_H_
#include <Maths/types.h>
namespace ksk::maths {
// modified from HOBAKv1
template <typename T, int Dim>
struct SVDResult {
  Matrix<T, Dim, Dim> U, V;
  Vector<T, Dim> S;
  SVDResult(const Matrix<T, Dim, Dim> &U, const Matrix<T, Dim, Dim> &V, const Vector<T, Dim> &S) : U(U), V(V), S(S) {}
  SVDResult(SVDResult&& other) noexcept : U(std::move(other.U)), V(std::move(other.V)), S(std::move(other.S)) {}
};
template <typename T, int Dim>
SVDResult<T, Dim> SVD(const Matrix<T, Dim, Dim> &A) {
  const Eigen::JacobiSVD<Matrix<T, Dim, Dim>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto U = svd.matrixU();
  auto V = svd.matrixV();
  auto S = svd.singularValues();
  Matrix<T, Dim, Dim> L = Matrix<T, Dim, Dim>::Identity();
  L(0, 0) = (U * V.transpose()).determinant();
  const Real detU = U.determinant();
  const Real detV = V.determinant();
  if (detU < 0.0 && detV > 0.0)
    U = U * L;
  else if (detU > 0.0 && detV < 0.0)
    V = V * L;
  S(0) = S(0) * L(0, 0);
  return {U, V, S};
}
}
#endif //KISEKI_MATHS_INCLUDE_MATHS_SVD_H_
