// ============================================================================
// src/cpu/cpu-projector.cc
// ============================================================================

#include <FluidSim/cpu/cpu-projector.h>
#include <iostream>

namespace fluid::cpu {

CPUProjector::CPUProjector(CPUFluidContext& ctx, ProjectorConfig config)
    : CPUModularSolver("cpu_projector"), config_(std::move(config))
{
    int nx = ctx.domain.resolution.x;
    int ny = ctx.domain.resolution.y;
    int nz = ctx.domain.resolution.z;

    auto fvm = std::make_unique<FvmSolver3D>(nx, ny, nz);
    fvm->setCompressedSolver(CompressedSolverMethod::CG, config_.preconditioner);
    projector_ = std::move(fvm);
}

void CPUProjector::solve(CPUFluidContext& ctx, Real dt) {
    std::cout << "  [CPUProjector] Building system... ";
    projector_->buildSystem(*ctx.u, *ctx.v, *ctx.w,
                            *ctx.fluidSdf, *ctx.colliderSdf, dt);
    std::cout << "Done." << std::endl;

    std::cout << "  [CPUProjector] Solving pressure... ";
    Real residual = projector_->solvePressure(*ctx.fluidSdf, ctx.pressure);
    if (residual > config_.tolerance)
        std::cerr << "Warning: projection residual is " << residual << std::endl;
    else
        std::cout << "residual = " << residual << std::endl;
    std::cout << "Done." << std::endl;

    std::cout << "  [CPUProjector] Projecting velocities... ";
    projector_->project(*ctx.u, *ctx.v, *ctx.w,
                        ctx.pressure, *ctx.fluidSdf, *ctx.colliderSdf, dt);

    // Apply collider boundary
    ctx.u->parallelForEach([&](int i, int j, int k) {
        if (i == 0 || i == ctx.u->width() - 1) {
            ctx.u->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos{ctx.u->indexToCoord(i, j, k)};
        if (ctx.colliderSdf->eval(pos) < 0.0) {
            Vec3d normal{normalize(ctx.colliderSdf->grad(pos))};
            ctx.u->at(i, j, k) -= ctx.u->at(i, j, k) * normal.x;
        }
    });
    ctx.v->parallelForEach([&](int i, int j, int k) {
        if (j == 0 || j == ctx.v->height() - 1) {
            ctx.v->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos = ctx.v->indexToCoord(i, j, k);
        if (ctx.colliderSdf->eval(pos) < 0.0) {
            Vec3d normal{normalize(ctx.colliderSdf->grad(pos))};
            ctx.v->at(i, j, k) -= ctx.v->at(i, j, k) * normal.y;
        }
    });
    ctx.w->parallelForEach([&](int i, int j, int k) {
        if (k == 0 || k == ctx.w->depth() - 1) {
            ctx.w->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos{ctx.w->indexToCoord(i, j, k)};
        if (ctx.colliderSdf->eval(pos) < 0.0) {
            Vec3d normal{normalize(ctx.colliderSdf->grad(pos))};
            ctx.w->at(i, j, k) -= ctx.w->at(i, j, k) * normal.z;
        }
    });
    std::cout << "Done." << std::endl;
}

void CPUProjector::configure(const core::Properties& props) {
    if (auto v = props.tryGet<int>("max_iters"))
        config_.maxIters = *v;
    if (auto v = props.tryGet<Real>("tolerance"))
        config_.tolerance = *v;
    if (auto v = props.tryGet<Real>("density"))
        config_.density = *v;
}

core::Properties CPUProjector::currentConfig() const {
    core::Properties p;
    p.set("max_iters", config_.maxIters);
    p.set("tolerance", config_.tolerance);
    p.set("density", config_.density);
    return p;
}

} // namespace fluid::cpu
