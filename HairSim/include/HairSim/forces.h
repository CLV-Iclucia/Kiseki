//
// Created by creeper on 23-8-10.
//

#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_FORCES_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_FORCES_H_

#include <HairSim/hair-dof.h>
#include <HairSim/hair-sim.h>
#include <HairSim/hair.h>
#include <HairSim/utils.h>
namespace hairsim {
class Force {
public:
  virtual void computeElementForce(const Hair &hair, Index e,
                                   Vec4d &f) const = 0;
  virtual void computeForce(const Hair &hair, VecXd &f) const = 0;
  virtual void computeStiffness(const Hair &hair, vector<Triplet<Real>> &J) const = 0;
  virtual ~Force() = default;
};

class BendingForce final : public Force {
public:
  void computeElementForce(const Hair &hair, Index e, Vec4d &f) const override;
  void computeElementStiffness(const Hair &hair, Index i, Index j,
                               Mat4d &H) const;
  void computeForce(const Hair &hair, VecXd &f) const override;
  void computeStiffness(const Hair &hair, vector<Triplet<Real>> &J) const override;
};

class TwistingForce final : public Force {
public:
  void computeElementForce(const Hair &hair, Index e, Vec4d &f) const override;
  void computeElementStiffness(const Hair &hair, Index i, Index j,
                               Mat4d &H) const;
  void computeForce(const Hair &hair, VecXd &f) const override;
  void computeStiffness(const Hair &hair, vector<Triplet<Real>> &J) const override;
};

class StretchingForce final : public Force {
public:
  void computeElementForce(const Hair &hair, Index e, Vec4d &f) const override;
  void computeElementStiffness(const Hair &hair, Index e,
                               Mat4d &H) const;
  void computeForce(const Hair &hair, VecXd &f) const override;
  void computeStiffness(const Hair &hair, vector<Triplet<Real>> &J) const override;
};

} // namespace hairsim

#endif // KISEKI_HAIRSIM_INCLUDE_HAIRSIM_FORCES_H_
