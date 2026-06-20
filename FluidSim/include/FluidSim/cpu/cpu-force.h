// ============================================================================
// include/FluidSim/cpu/cpu-force.h
// CPUForceApplicator: 施加外力（重力等）
// ============================================================================
#pragma once

#include <FluidSim/cpu/cpu-solver.h>

namespace fluid::cpu {

struct ForceConfig {
    Vec3d gravity{0.0, -9.8, 0.0};
};

class CPUForceApplicator : public CPUModularSolver {
public:
    explicit CPUForceApplicator(CPUFluidContext& ctx, ForceConfig config = {})
        : CPUModularSolver("cpu_force"), config_(config)
    {
        (void)ctx;
    }

    void solve(CPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    ForceConfig config_;
};

} // namespace fluid::cpu
