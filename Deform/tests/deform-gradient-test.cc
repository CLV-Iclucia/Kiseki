//
// Created by creeper on 9/13/24.
//
#include <Deform/strain-energy-density.h>
using namespace ksk::deform;

Matrix<Real, 3, 3> tetDs(const Vector<Real, 12> &x) {
  Matrix<Real, 3, 3> Ds;
  Ds.col(0) = x.segment<3>(3) - x.segment<3>(0);
  Ds.col(1) = x.segment<3>(6) - x.segment<3>(0);
  Ds.col(2) = x.segment<3>(9) - x.segment<3>(0);
  return Ds;
}

Vector<Real, 12> tetEnergyGradient(const StableNeoHookean<Real> &energy,
                                   const DeformationGradient<Real, 3> &dg) {
  auto gradient_F = energy.computeEnergyGradient(dg);
  auto p_F_p_x = dg.gradient();
  Vector<Real, 12> gradient_x = p_F_p_x.transpose() * maths::vectorize(gradient_F);
  return gradient_x;
}

int main() {
  auto energy = std::make_unique<StableNeoHookean<Real>>(ElasticityParameters<Real>{1e7, 0.45});
  Vector<Real, 3> a(1.75, 1, 2);
  Vector<Real, 3> b(1.75, 2, 1);
  Vector<Real, 3> c(0.75, 2, 2);
  Vector<Real, 3> d(1.75, 2, 2);
  Vector<Real, 12> x;
  x << a, b, c, d;
  auto dg = DeformationGradient<Real, 3>(tetDs(x));
  for (int i = 0; i < 12; i++)
    x(i) += 0.2;
  auto Ds = tetDs(x);
  dg.updateCurrentConfig(Ds);
  auto symbolicGradient = energy->computeEnergyGradient(dg);
  Matrix<Real, 3, 3> numericalGradient;
  Real dx = 1e-5;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      auto Ds_dx = Ds;
      Ds_dx(j, i) += dx;
      dg.updateCurrentConfig(Ds_dx);
      auto E_plus = energy->computeEnergyDensity(dg);
      Ds_dx(j, i) -= 2 * dx;
      dg.updateCurrentConfig(Ds_dx);
      auto E_minus = energy->computeEnergyDensity(dg);
      numericalGradient(j, i) = (E_plus - E_minus) / (2 * dx);
    }
  }
  if ((symbolicGradient - numericalGradient).norm() > 1e-3) {
    std::cerr << "Symbolic and numerical gradients do not match for energy\n";
    std::cerr << "Symbolic gradient:\n" << symbolicGradient << std::endl;
    std::cerr << "Numerical gradient:\n" << numericalGradient << std::endl;
    exit(1);
  }

  dg.updateCurrentConfig(Ds);
  auto symbolicGradientWrtX = tetEnergyGradient(*energy, dg);
  Vector<Real, 12> numericalGradientWrtX;
  for (int i = 0; i < 12; i++) {
    auto x_plus = x;
    x_plus(i) += dx;
    dg.updateCurrentConfig(tetDs(x_plus));
    auto E_plus = energy->computeEnergyDensity(dg);
    x_plus(i) -= 2 * dx;
    dg.updateCurrentConfig(tetDs(x_plus));
    auto E_minus = energy->computeEnergyDensity(dg);
    numericalGradientWrtX(i) = (E_plus - E_minus) / (2 * dx);
  }
  if ((symbolicGradientWrtX - numericalGradientWrtX).norm() > 1e-3) {
    std::cerr << "Symbolic and numerical gradients do not match\n";
    std::cerr << "Symbolic gradient:\n" << symbolicGradientWrtX.transpose() << std::endl;
    std::cerr << "Numerical gradient:\n" << numericalGradientWrtX.transpose() << std::endl;
    exit(1);
  }
}