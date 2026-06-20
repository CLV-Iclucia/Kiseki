// ============================================================================
// include/FluidSim/cpu/cpu-reconstructor.h
// CPUReconstructor: 粒子 → fluid SDF 重建 + smooth
// 内部复用现有 NaiveReconstructor
// ============================================================================
#pragma once

#include <FluidSim/cpu/cpu-solver.h>
#include <FluidSim/cpu/sdf.h>
#include <memory>

namespace fluid::cpu {

struct ReconstructorConfig {
    ReconstructorMethod method = ReconstructorMethod::Naive;
    Real particleRadius = 0.8;  // 乘以 gridSpacing
    int smoothIters = 5;
    int extrapolateIters = 10;
};

class CPUReconstructor : public CPUModularSolver {
public:
    CPUReconstructor(CPUFluidContext& ctx, ReconstructorConfig config);

    void solve(CPUFluidContext& ctx, Real dt) override;

    void configure(const core::Properties& props) override;

private:
    ReconstructorConfig config_;
    std::unique_ptr<ParticleSystemReconstructor<Real, 3>> reconstructor_;

    void extrapolateFluidSdf(CPUFluidContext& ctx, int iters);
    void smoothFluidSurface(CPUFluidContext& ctx, int iters);
};

} // namespace fluid::cpu
