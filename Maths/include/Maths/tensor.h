//
// Created by creeper on 5/25/24.
//

#ifndef KISEKI_MATHS_INCLUDE_MATHS_TENSOR_H_
#define KISEKI_MATHS_INCLUDE_MATHS_TENSOR_H_
#include <Maths/types.h>
namespace ksk::maths {
template<typename T, int N, int M>
auto vectorize(const Matrix<T, N, M> &A) {
  return A.reshaped(N * M, 1).eval();
}
//template<typename T, int N>
//using FourthOrderTensor = Matrix<T, N * N, N * N>;
//template<typename T, int N>
//auto tensorProduct(const Vector<T, N> &a, const Vector<T, N> &b) {
//  return a * b.transpose();
//}
//template<typename T, int N, int M>
//using ThirdOrderTensor = Matrix<T, N * N, M>;
//
//template <typename T, int N, int M>
//auto submatrix(ThirdOrderTensor<T, N, M> &tensor, int i) {
//  assert(i < M);
//  return Eigen::Map<Matrix<T, N, N>>(tensor.data() + i * N * N);
//}
//template <typename T, int N, int M>
//auto submatrix(const ThirdOrderTensor<T, N, M> &tensor, int i) {
//  assert(i < M);
//  return Eigen::Map<const Matrix<T, N, N>>(tensor.data() + i * N * N);
//}
//template<typename T, int N, int M>
//auto thirdOrderDoubleContract(const ThirdOrderTensor<T, N, M> &tensor, const Matrix<T, N, N> &A) {
//  return tensor.transpose() * vectorize(A);
//}
//template<typename T, int N>
//auto fourthOrderDoubleContract(const FourthOrderTensor<T, N> &tensor, const Matrix<T, N, N> &A) {
//  return tensor.transpose() * vectorize(A);
//}
template<typename T>
Matrix<T, 3, 3> determinantGradient(const Matrix<T, 3, 3> &F) {
  Matrix<T, 3, 3> result;
  result.col(0) = F.col(1).cross(F.col(2));
  result.col(1) = F.col(2).cross(F.col(0));
  result.col(2) = F.col(0).cross(F.col(1));
  return result;
}
}
#endif //KISEKI_MATHS_INCLUDE_MATHS_TENSOR_H_
