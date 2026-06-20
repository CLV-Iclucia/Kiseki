// ============================================================================
// include/FluidSim/cpu/cpu-projector.h
// CPUProjector: 压力投影的模块化 Solver
// 内部复用现有 FvmSolver3D / CgSolver3D
// ============================================================================
#pragma once

#include <FluidSim/cpu/cpu-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <memory>

namespace fluid::cpu {

struct ProjectorConfig {
    PreconditionerMethod preconditioner = PreconditionerMethod::ModifiedIncompleteCholesky;
    int maxIters = 300;
    Real tolerance = 1e-4;
    Real density = 1000.0;
};

class CPUProjector : public CPUModularSolver {
public:
    CPUProjector(CPUFluidContext& ctx, ProjectorConfig config);

    void solve(CPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;
    core::Properties currentConfig() const override;

private:
    ProjectorConfig config_;
    std::unique_ptr<ProjectionSolver3D> projector_;
};

} // namespace fluid::cpu
