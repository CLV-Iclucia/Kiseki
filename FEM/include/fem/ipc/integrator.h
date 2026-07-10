//
// Created by creeper on 10/22/24.
//

#pragma once
#include <Core/debug.h>
#include <fem/integrator.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/collision-pair.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/gipc/barrier.h>
#include <Maths/block-linear-solver.h>
#include <functional>

namespace ksk::fem {

class IpcIntegrator : public Integrator {
  public:
    struct Config {
      Real dHat = 1e-3;
      Real eps = 1e-2;
      Real contactStiffness = 1e10;
      Real stepSizeScale = 0.9;
      REFLECT(dHat, eps, contactStiffness, stepSizeScale);
    } config;

    explicit IpcIntegrator(System &system, const Config &config)
      : Integrator(system), config(config), barrier_(config.dHat) {
      collisionDetector = std::make_unique<ipc::CollisionDetector>(system);
    }

    void step(Real dt) override;
    std::unique_ptr<maths::BlockLinearSolver> solver;
    static std::unique_ptr<Integrator> create(System &system, const core::JsonNode &json);

    /// Optional callback invoked after each Newton iteration (for debug visualization)
    std::function<void(int iter)> onNewtonIter;

  protected:

    [[nodiscard]] Real barrierEnergy() const;

    [[nodiscard]] maths::BlockVector<3> barrierEnergyGradient() const;

    virtual void velocityUpdate(const maths::BlockVector<3> &x_t, Real h) const = 0;
    [[nodiscard]] virtual Real incrementalPotentialKinematicEnergy(const maths::BlockVector<3> &x_t, Real h) const = 0;
    [[nodiscard]] virtual maths::BlockVector<3> incrementalPotentialKinematicEnergyGradient(const maths::BlockVector<3> &x_t, Real h) const = 0;

    [[nodiscard]] Real barrierAugmentedPotentialEnergy() const {
      return system().potentialEnergy() + barrierEnergy();
    }

    [[nodiscard]] maths::BlockVector<3> barrierAugmentedPotentialEnergyGradient() const {
      auto grad = system().deformationEnergyGradient();
      auto barrierGrad = barrierEnergyGradient();
      grad += barrierGrad;
      return grad;
    }

    [[nodiscard]] Real barrierAugmentedIncrementalPotentialEnergy(const maths::BlockVector<3> &x_t, Real h) const {
      return incrementalPotentialKinematicEnergy(x_t, h) + h * h * barrierAugmentedPotentialEnergy();
    }

    [[nodiscard]] maths::BlockVector<3> barrierAugmentedIncrementalPotentialEnergyGradient(const maths::BlockVector<3> &x_t, Real h) const {
      auto kinGrad = incrementalPotentialKinematicEnergyGradient(x_t, h);
      auto potGrad = barrierAugmentedPotentialEnergyGradient();
      potGrad *= (h * h);
      kinGrad += potGrad;
      return kinGrad;
    }

    void updateCandidateSolution(const maths::BlockVector<3> &x) {
      system().updateCurrentConfig(x);
      refreshActiveConstraintPairs();
    }

    [[nodiscard]] Real computeStepSizeUpperBound(const maths::BlockVector<3> &p) const {
      collisionDetector->updateBVHs(p, 1.0);
      auto t = collisionDetector->detect(p);
      return t.value_or(1.0);
    }

    [[nodiscard]] maths::BlockSparseMatrix<3> spdProjectHessian(Real h) const;
    
    /// 从 collision pair 重建 active constraint pair 集合
    void refreshActiveConstraintPairs();
    
    /// Newton precompute 步骤：构建 collision pair
    void precomputeCollisionPairs(const maths::BlockVector<3>& p, Real alpha);
    
    void computeVertexTriangleCollisionPairs(const maths::BlockVector<3>& p, Real alpha);
    void computeEdgeEdgeCollisionPairs(const maths::BlockVector<3>& p, Real alpha);


    ipc::CollisionPairSet collisionPairs;    // broad phase 候选层
    ipc::ConstraintPairSet constraintPairs;  // active 约束层
    bool hasInitializedActiveConstraints = false;

    
    std::unique_ptr<ipc::CollisionDetector> collisionDetector{};
    ipc::gipc::Barrier barrier_;
    maths::BlockVector<3> x_prev;
};
}

