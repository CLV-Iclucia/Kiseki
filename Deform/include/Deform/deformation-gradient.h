//
// Created by creeper on 6/3/24.
//

#ifndef KISEKI_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
#define KISEKI_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
#include <Deform/types.h>
#include <Maths/svd.h>
namespace ksk::deform {
template<typename T, int Dim>
struct DeformationGradient {
  DeformationGradient() : m_Dm_inverse(Matrix<T, Dim, Dim>::Identity()) {}
  explicit DeformationGradient(const Matrix<T, Dim, Dim> &local_X) {
    m_Dm_inverse = local_X.inverse();
    m_F = Matrix<T, Dim, Dim>::Identity();
    m_U = Matrix<T, Dim, Dim>::Identity();
    m_V = Matrix<T, Dim, Dim>::Identity();
    m_Sigma = Vector<T, Dim>::Ones();
  }
  DeformationGradient(const Matrix<T, Dim, Dim>& local_x, const Matrix<T, Dim, Dim>& local_X) {
    m_Dm_inverse = local_X.inverse();
    updateCurrentConfig(local_x);
  }
  void updateCurrentConfig(const Matrix<T, Dim, Dim> &local_x) {
    m_F = local_x * m_Dm_inverse;
    computeSVD();
  }
  Matrix<T, Dim * Dim, Dim * (Dim + 1)> gradient() const {
    if constexpr (Dim == 3) {
      Matrix<T, Dim * Dim, Dim * (Dim + 1)> result;
      result.setZero();
      const auto &r0 = m_Dm_inverse.row(0);
      const auto &r1 = m_Dm_inverse.row(1);
      const auto &r2 = m_Dm_inverse.row(2);
      auto s0 = m_Dm_inverse.col(0).sum();
      auto s1 = m_Dm_inverse.col(1).sum();
      auto s2 = m_Dm_inverse.col(2).sum();
      result.col(0).reshaped(3, 3).row(0) = Vector<T, Dim>(-s0, -s1, -s2);
      result.col(1).reshaped(3, 3).row(1) = Vector<T, Dim>(-s0, -s1, -s2);
      result.col(2).reshaped(3, 3).row(2) = Vector<T, Dim>(-s0, -s1, -s2);
      result.col(3).reshaped(3, 3).row(0) = r0;
      result.col(6).reshaped(3, 3).row(0) = r1;
      result.col(9).reshaped(3, 3).row(0) = r2;
      result.col(4).reshaped(3, 3).row(1) = r0;
      result.col(7).reshaped(3, 3).row(1) = r1;
      result.col(10).reshaped(3, 3).row(1) = r2;
      result.col(5).reshaped(3, 3).row(2) = r0;
      result.col(8).reshaped(3, 3).row(2) = r1;
      result.col(11).reshaped(3, 3).row(2) = r2;
      return result;
    } else
      throw std::runtime_error("Sorry, gradient of deformation gradient is not implemented for dimension other than 3");
  }
  const Matrix<T, Dim, Dim> &U() const {
    return m_U;
  }
  const Matrix<T, Dim, Dim> &V() const {
    return m_V;
  }
  // read-only
  const Matrix<T, Dim, Dim> &F() const {
    return m_F;
  }
  Matrix<T, Dim, Dim> R() const {
    return m_U * m_V.transpose();
  }
  const Vector<T, Dim> &Sigma() const {
    return m_Sigma;
  }
  [[nodiscard]] Real Sigma(int i) const {
    return m_Sigma(i);
  }
  Matrix<T, Dim, Dim> S() const {
    return m_V * m_Sigma.asDiagonal() * m_V.transpose();
  }
 private:
  void computeSVD() {
    auto&& result = maths::SVD(m_F);
    m_U = result.U;
    m_V = result.V;
    m_Sigma = result.S;
  }
  Matrix<Real, Dim, Dim> m_F, m_U, m_V, m_Dm_inverse;
  Vector<Real, Dim> m_Sigma;
};


}
#endif //KISEKI_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
