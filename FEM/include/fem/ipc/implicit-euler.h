//
// Created by creeper on 5/28/24.
//

#pragma once
#include <fem/integrator.h>
#include <fem/ipc/integrator.h>

namespace ksk::fem {

struct IpcImplicitEuler final : public IpcIntegrator {
  explicit IpcImplicitEuler(System &system, const Config &config = {})
    : IpcIntegrator(system, config) {
  }

private:
  [[nodiscard]] Real incrementalPotentialKinematicEnergy(const maths::BlockVector<3> &x_t, Real h) const override;

  void velocityUpdate(const maths::BlockVector<3> &x_t, Real h) const override {
    maths::BlockVector<3> diff = system().x;
    diff -= x_t;
    diff *= (1.0 / h);
    system().xdot = diff;
  }

  [[nodiscard]] maths::BlockVector<3> incrementalPotentialKinematicEnergyGradient(const maths::BlockVector<3> &x_t, Real h) const override;

  friend maths::BlockVector<3> symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const maths::BlockVector<3> &x_t, Real h);
  friend maths::BlockVector<3> numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const maths::BlockVector<3> &x_t, Real h);
};

}