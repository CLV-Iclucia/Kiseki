// ============================================================================
// src/cpu/cpu-force.cc
// ============================================================================

#include <FluidSim/cpu/cpu-force.h>
#include <iostream>

namespace fluid::cpu {

void CPUForceApplicator::solve(CPUFluidContext& ctx, Real dt) {
    std::cout << "  [CPUForceApplicator] Applying forces... ";

    // 应用重力到 v (y 分量)
    if (config_.gravity.y != 0.0) {
        ctx.v->forEach([&](int i, int j, int k) {
            if (j > 0 && j < ctx.v->height() - 1)
                ctx.v->at(i, j, k) += config_.gravity.y * dt;
        });
    }

    // 应用重力到 u (x 分量)
    if (config_.gravity.x != 0.0) {
        ctx.u->forEach([&](int i, int j, int k) {
            if (i > 0 && i < ctx.u->width() - 1)
                ctx.u->at(i, j, k) += config_.gravity.x * dt;
        });
    }

    // 应用重力到 w (z 分量)
    if (config_.gravity.z != 0.0) {
        ctx.w->forEach([&](int i, int j, int k) {
            if (k > 0 && k < ctx.w->depth() - 1)
                ctx.w->at(i, j, k) += config_.gravity.z * dt;
        });
    }

    std::cout << "Done." << std::endl;
}

void CPUForceApplicator::configure(const core::Properties& props) {
    if (auto v = props.tryGet<Real>("gravity_x"))
        config_.gravity.x = *v;
    if (auto v = props.tryGet<Real>("gravity_y"))
        config_.gravity.y = *v;
    if (auto v = props.tryGet<Real>("gravity_z"))
        config_.gravity.z = *v;
}

} // namespace fluid::cpu
