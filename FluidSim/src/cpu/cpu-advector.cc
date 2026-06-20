// ============================================================================
// src/cpu/cpu-advector.cc
// ============================================================================

#include <FluidSim/cpu/cpu-advector.h>
#include <iostream>
#include <span>

namespace fluid::cpu {

// ============================================================================
// CPUAdvector (共享实现)
// ============================================================================

CPUAdvector::CPUAdvector(CPUFluidContext& ctx, const AdvectorConfig& config) {
    int nParticles = static_cast<int>(ctx.positions.size());
    Vec3i res = ctx.domain.resolution;
    Real w = ctx.domain.size.x;
    Real h = ctx.domain.size.y;
    Real d = ctx.domain.size.z;

    switch (config.method) {
    case AdvectionMethod::PIC:
        advector_ = std::make_unique<PicAdvector3D>(nParticles, res, w, h, d);
        break;
    case AdvectionMethod::FLIP:
    case AdvectionMethod::APIC:  // APIC fallback to FLIP for now
        advector_ = std::make_unique<FlipAdvectionSolver3D>(nParticles, res, w, h, d);
        break;
    }
}

// ============================================================================
// CPUAdvectorP2G
// ============================================================================

void CPUAdvectorP2G::solve(CPUFluidContext& ctx, Real dt) {
    // 1. 先做 advection（RK3 前推粒子位置并处理碰撞）
    std::cout << "  [CPUAdvectorP2G] Advect... ";
    impl_->impl().advect(std::span(ctx.positions),
                         *ctx.u, *ctx.v, *ctx.w, *ctx.colliderSdf, dt);
    std::cout << "Done." << std::endl;

    // 2. 清空网格和标记
    ctx.uValid->fill(0);
    ctx.vValid->fill(0);
    ctx.wValid->fill(0);
    ctx.uValidBuf->fill(0);
    ctx.vValidBuf->fill(0);
    ctx.wValidBuf->fill(0);
    ctx.u->fill(0);
    ctx.v->fill(0);
    ctx.w->fill(0);
    ctx.uBuf->fill(0);
    ctx.vBuf->fill(0);
    ctx.wBuf->fill(0);
    ctx.uw.fill(0);
    ctx.vw.fill(0);
    ctx.ww.fill(0);
    ctx.pressure.fill(0);

    // 3. P2G scatter
    std::cout << "  [CPUAdvectorP2G] P2G scatter... ";
    impl_->impl().solveP2G(std::span(ctx.positions),
                           *ctx.u, *ctx.v, *ctx.w, *ctx.colliderSdf,
                           ctx.uw, ctx.vw, ctx.ww,
                           *ctx.uValid, *ctx.vValid, *ctx.wValid, dt);

    // 4. Apply Dirichlet boundary conditions
    // Boundaries: zero velocity at domain walls
    for (int j = 0; j < ctx.u->height(); j++) {
        for (int k = 0; k < ctx.u->depth(); k++) {
            ctx.u->at(0, j, k) = 0.0;
            ctx.uValid->at(0, j, k) = 1;
            ctx.u->at(ctx.u->width() - 1, j, k) = 0.0;
            ctx.uValid->at(ctx.u->width() - 1, j, k) = 1;
        }
    }
    for (int i = 0; i < ctx.v->width(); i++) {
        for (int k = 0; k < ctx.v->depth(); k++) {
            ctx.v->at(i, 0, k) = 0.0;
            ctx.vValid->at(i, 0, k) = 1;
            ctx.v->at(i, ctx.v->height() - 1, k) = 0.0;
            ctx.vValid->at(i, ctx.v->height() - 1, k) = 1;
        }
    }
    for (int i = 0; i < ctx.w->width(); i++) {
        for (int j = 0; j < ctx.w->height(); j++) {
            ctx.w->at(i, j, 0) = 0.0;
            ctx.wValid->at(i, j, 0) = 1;
            ctx.w->at(i, j, ctx.w->depth() - 1) = 0.0;
            ctx.wValid->at(i, j, ctx.w->depth() - 1) = 1;
        }
    }
    std::cout << "Done." << std::endl;
}

void CPUAdvectorP2G::configure(const core::Properties& props) {
    // 运行时配置更新（如 flip_blend）可在此实现
    (void)props;
}

// ============================================================================
// CPUAdvectorG2P
// ============================================================================

void CPUAdvectorG2P::solve(CPUFluidContext& ctx, Real dt) {
    std::cout << "  [CPUAdvectorG2P] G2P gather... ";
    impl_->impl().solveG2P(std::span(ctx.positions),
                           *ctx.u, *ctx.v, *ctx.w, *ctx.colliderSdf, dt);
    std::cout << "Done." << std::endl;
}

} // namespace fluid::cpu
