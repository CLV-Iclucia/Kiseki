//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/system.h>

namespace ksk::fem {

Real IpcImplicitEuler::incrementalPotentialKinematicEnergy(const maths::BlockVector<3> &x_t, Real h) const {
  // x_hat = x_t + h*v + h²*a
  maths::BlockVector<3> x_hat = x_t;
  x_hat.axpy(h, system().xdot);
  auto a = system().computeAcceleration();
  x_hat.axpy(h * h, a);

  maths::BlockVector<3> diff = system().x;
  diff -= x_hat;

  maths::BlockVector<3> M_diff(diff.numBlocks());
  system().blockMass().apply(diff, M_diff);

  return 0.5 * diff.dot(M_diff);
}

maths::BlockVector<3> IpcImplicitEuler::incrementalPotentialKinematicEnergyGradient(const maths::BlockVector<3> &x_t, Real h) const {
  // x_hat = x_t + h*v + h²*a
  maths::BlockVector<3> x_hat = x_t;
  x_hat.axpy(h, system().xdot);
  auto a = system().computeAcceleration();
  x_hat.axpy(h * h, a);
  // grad = M * (x - x_hat)
  maths::BlockVector<3> diff = system().x;
  diff -= x_hat;
  maths::BlockVector<3> result(diff.numBlocks());
  system().blockMass().apply(diff, result);
  return result;
}

maths::BlockVector<3> symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const maths::BlockVector<3> &x_t, Real h) {
  return euler.barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
}

maths::BlockVector<3> numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const maths::BlockVector<3> &x_t, Real h) {
  int n = euler.system().dof();
  maths::BlockVector<3> grad(euler.system().x.numBlocks());
  grad.setZero();
  Real dx = 1e-5;
  maths::BlockVector<3> current = euler.system().x;
  for (int i = 0; i < n; i++) {
    maths::BlockVector<3> x_plus = current;
    x_plus.data()[i] += dx;
    euler.updateCandidateSolution(x_plus);
    Real E_plus = euler.barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    maths::BlockVector<3> x_minus = current;
    x_minus.data()[i] -= dx;
    euler.updateCandidateSolution(x_minus);
    Real E_minus = euler.barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    grad.data()[i] = (E_plus - E_minus) / (2 * dx);
  }
  return grad;
}

}
