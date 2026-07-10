#pragma once

#include <HairSim/rod.h>

#include <Eigen/Core>

namespace ksk::hairsim {

struct RodAssembly {
  RodEnergyComponents& energy;
  Eigen::VectorXd& gradient;
  Eigen::MatrixXd& hessian;
};

class StretchingEnergy {
 public:
  StretchingEnergy(const RodState& state, const RodRestState& rest,
                   const RodMaterial& material);

  void accumulate(RodAssembly assembly) const;

 private:
  const RodState& state_;
  const RodRestState& rest_;
  const RodMaterial& material_;
};

class BendingEnergy {
 public:
  BendingEnergy(const RodState& state, const RodRestState& rest,
                const RodMaterial& material);

  void accumulate(RodAssembly assembly) const;

 private:
  const RodState& state_;
  const RodRestState& rest_;
  const RodMaterial& material_;
};

class TwistingEnergy {
 public:
  TwistingEnergy(const RodState& state, const RodRestState& rest,
                 const RodMaterial& material);

  void accumulate(RodAssembly assembly) const;

 private:
  const RodState& state_;
  const RodRestState& rest_;
  const RodMaterial& material_;
};

}  // namespace ksk::hairsim
