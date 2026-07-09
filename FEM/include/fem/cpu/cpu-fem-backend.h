// ============================================================================
// FEM/include/fem/cpu/cpu-fem-backend.h
// CPUFEMBackend — wraps existing System + Integrator (zero functional change)
// ============================================================================
#pragma once

#include <fem/fem-backend.h>
#include <fem/system.h>
#include <fem/integrator.h>
#include <Maths/block-linear-solver.h>
#include <memory>

namespace sim::fem {

class CPUFEMBackend : public FEMBackend {
public:
    CPUFEMBackend() = default;

    void initialize(const FEMScene& scene) override;
    void step(Real dt) override;
    void readback(FEMFrame& out) override;

    /// Direct access to the internal system (for testing / debug)
    System& getSystem() { return system_; }
    const System& getSystem() const { return system_; }

protected:
    System system_;
    std::unique_ptr<Integrator> integrator_;

    /// Track Newton iterations for readback
    int lastNewtonIters_ = 0;

    /// Build System from FEMScene
    void buildSystem(const FEMScene& scene);

    /// Build Integrator from FEMScene parameters
    void buildIntegrator(const FEMScene& scene);

    /// Create the linear solver injected into the Newton loop.
    /// Overridden by GPUFEMBackend to inject a GPU-resident PCG solver.
    virtual std::unique_ptr<maths::BlockLinearSolver>
    makeLinearSolver(const FEMScene& scene);
};

} // namespace sim::fem
