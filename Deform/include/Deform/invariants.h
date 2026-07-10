//
// Created by creeper on 5/23/24.
//

#pragma once
#include <Deform/types.h>
#include <Deform/deformation-gradient.h>
#include <Core/properties.h>
#include <Maths/tensor.h>
#include <Maths/linalg-utils.h>
namespace ksk::deform {
template<typename T>
struct EigenMatrixSigmas : core::NonCopyable {
  std::array<Matrix<T, 3, 3>, 3> eigenMatrixSigmas;
  EigenMatrixSigmas() {
    eigenMatrixSigmas[0].setZero();
    eigenMatrixSigmas[0](0, 1) = -1.0;
    eigenMatrixSigmas[0](1, 0) = 1.0;
    eigenMatrixSigmas[1].setZero();
    eigenMatrixSigmas[1](1, 2) = 1.0;
    eigenMatrixSigmas[1](2, 1) = -1.0;
    eigenMatrixSigmas[2].setZero();
    eigenMatrixSigmas[2](0, 2) = 1.0;
    eigenMatrixSigmas[2](2, 0) = -1.0;
  }
  [[nodiscard]] const auto &S(int i) const {
    return eigenMatrixSigmas[i];
  }
};

template<typename T>
const EigenMatrixSigmas<T> &eigenMatrixSigmas() {
  static EigenMatrixSigmas<T> instance;
  return instance;
}

template<typename T>
Vector<T, 3> isotropicInvariants(const DeformationGradient<T, 3> &dg) {
  return {dg.S().trace(), (dg.F().transpose() * dg.F()).trace(), dg.F().determinant()};
}

template<typename T>
Matrix<T, 3, 3> gradientIi(const DeformationGradient<T, 3> &dg) {
  return dg.R();
}

template<typename T>
Matrix<T, 3, 3> gradientIii(const DeformationGradient<T, 3> &dg) {
  return 2.0 * dg.F();
}

template<typename T>
Matrix<T, 3, 3> gradientIiii(const DeformationGradient<T, 3> &dg) {
  return maths::determinantGradient(dg.F());
}
template<typename T>
Matrix<T, 9, 9> hessianIi(const DeformationGradient<T, 3> &dg) {
  std::array<Real, 3> lambda{2.0 / (dg.Sigma(0) + dg.Sigma(1)), 2.0 / (dg.Sigma(1) + dg.Sigma(2)),
                             2.0 / (dg.Sigma(0) + dg.Sigma(2))};
  Matrix<T, 9, 9> pRpF;
  pRpF.setZero();
  for (int i = 0; i < 3; i++) {
    Matrix<Real, 3, 3> Q = dg.U() * eigenMatrixSigmas<T>().S(i) * dg.V().transpose();
    pRpF += 0.5 * lambda[i] * maths::vectorize(Q) * maths::vectorize(Q).transpose();
  }
  return pRpF;
}
template<typename T>
Matrix<T, 9, 9> hessianIii() {
  return 2 * Matrix<T, 9, 9>::Identity();
}
template<typename T>
Matrix<T, 9, 9> hessianIiii(const DeformationGradient<T, 3> &dg) {
  Matrix<T, 9, 9> result{};
  result.setZero();
  result.template block<3, 3>(3, 0) = maths::skewt(Vector<T, 3>(dg.F().col(2)));
  result.template block<3, 3>(6, 0) = -maths::skewt(Vector<T, 3>(dg.F().col(1)));
  result.template block<3, 3>(0, 3) = -maths::skewt(Vector<T, 3>(dg.F().col(2)));
  result.template block<3, 3>(6, 3) = maths::skewt(Vector<T, 3>(dg.F().col(0)));
  result.template block<3, 3>(0, 6) = maths::skewt(Vector<T, 3>(dg.F().col(1)));
  result.template block<3, 3>(3, 6) = -maths::skewt(Vector<T, 3>(dg.F().col(0)));
  return result;
}

}
