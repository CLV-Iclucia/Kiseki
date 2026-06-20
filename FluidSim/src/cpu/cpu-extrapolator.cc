// ============================================================================
// src/cpu/cpu-extrapolator.cc
// ============================================================================

#include <FluidSim/cpu/cpu-extrapolator.h>
#include <iostream>

namespace fluid::cpu {

void CPUExtrapolator::solve(CPUFluidContext& ctx, Real dt) {
    (void)dt;
    std::cout << "  [CPUExtrapolator] Extrapolating velocities... ";
    extrapolate(ctx.u, ctx.uBuf, ctx.uValid, ctx.uValidBuf, config_.iters);
    extrapolate(ctx.v, ctx.vBuf, ctx.vValid, ctx.vValidBuf, config_.iters);
    extrapolate(ctx.w, ctx.wBuf, ctx.wValid, ctx.wValidBuf, config_.iters);
    std::cout << "Done." << std::endl;
}

void CPUExtrapolator::configure(const core::Properties& props) {
    if (auto v = props.tryGet<int>("iters"))
        config_.iters = *v;
}

} // namespace fluid::cpu
