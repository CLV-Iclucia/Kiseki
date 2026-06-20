// ============================================================================
// src/cpu/cpu-reconstructor.cc
// ============================================================================

#include <FluidSim/cpu/cpu-reconstructor.h>
#include <iostream>
#include <span>

namespace fluid::cpu {

CPUReconstructor::CPUReconstructor(CPUFluidContext& ctx, ReconstructorConfig config)
    : CPUModularSolver("cpu_reconstructor"), config_(std::move(config))
{
    int nParticles = static_cast<int>(ctx.positions.size());
    int nx = ctx.domain.resolution.x;
    int ny = ctx.domain.resolution.y;
    int nz = ctx.domain.resolution.z;
    Vec3d size = ctx.domain.size;

    reconstructor_ = std::make_unique<NaiveReconstructor<Real, 3>>(
        nParticles, nx, ny, nz, Vector<Real, 3>(size.x, size.y, size.z));
}

void CPUReconstructor::solve(CPUFluidContext& ctx, Real dt) {
    (void)dt;

    std::cout << "  [CPUReconstructor] Reconstructing surface... ";
    Real radius = config_.particleRadius * ctx.gridSpacing;
    reconstructor_->reconstruct(
        std::span<Vec3d>(ctx.positions),
        radius, *ctx.fluidSdf, *ctx.sdfValid);
    std::cout << "Done." << std::endl;

    std::cout << "  [CPUReconstructor] Extrapolating SDF... ";
    extrapolateFluidSdf(ctx, config_.extrapolateIters);
    std::cout << "Done." << std::endl;

    std::cout << "  [CPUReconstructor] Smoothing SDF... ";
    smoothFluidSurface(ctx, config_.smoothIters);
    std::cout << "Done." << std::endl;
}

void CPUReconstructor::extrapolateFluidSdf(CPUFluidContext& ctx, int iters) {
    for (int iter = 0; iter < iters; iter++) {
        ctx.sdfValidBuf->fill(false);
        ctx.fluidSdf->grid.forEach([&](int i, int j, int k) {
            if (ctx.sdfValid->at(i, j, k)) {
                ctx.sdfValidBuf->at(i, j, k) = true;
                return;
            }
            Real sum{0.0};
            int count{0};
            if (i > 0 && ctx.sdfValid->at(i - 1, j, k)) {
                sum += ctx.fluidSdf->grid(i - 1, j, k);
                count++;
            }
            if (i < ctx.fluidSdf->width() - 1 && ctx.sdfValid->at(i + 1, j, k)) {
                sum += ctx.fluidSdf->grid(i + 1, j, k);
                count++;
            }
            if (j > 0 && ctx.sdfValid->at(i, j - 1, k)) {
                sum += ctx.fluidSdf->grid(i, j - 1, k);
                count++;
            }
            if (j < ctx.fluidSdf->height() - 1 && ctx.sdfValid->at(i, j + 1, k)) {
                sum += ctx.fluidSdf->grid(i, j + 1, k);
                count++;
            }
            if (k > 0 && ctx.sdfValid->at(i, j, k - 1)) {
                sum += ctx.fluidSdf->grid(i, j, k - 1);
                count++;
            }
            if (k < ctx.fluidSdf->depth() - 1 && ctx.sdfValid->at(i, j, k + 1)) {
                sum += ctx.fluidSdf->grid(i, j, k + 1);
                count++;
            }
            if (count > 0) {
                ctx.fluidSdfBuf->grid(i, j, k) = sum / count;
                ctx.sdfValidBuf->at(i, j, k) = true;
            } else {
                ctx.sdfValidBuf->at(i, j, k) = false;
            }
        });
        std::swap(ctx.fluidSdf, ctx.fluidSdfBuf);
        std::swap(ctx.sdfValid, ctx.sdfValidBuf);
    }
}

void CPUReconstructor::smoothFluidSurface(CPUFluidContext& ctx, int iters) {
    for (int iter = 0; iter < iters; iter++) {
        ctx.fluidSdfBuf->grid.forEach([&](int i, int j, int k) {
            Real sum = 0;
            int count = 0;
            if (i > 0) {
                sum += ctx.fluidSdf->grid(i - 1, j, k);
                count++;
            }
            if (i < ctx.fluidSdf->width() - 1) {
                sum += ctx.fluidSdf->grid(i + 1, j, k);
                count++;
            }
            if (j > 0) {
                sum += ctx.fluidSdf->grid(i, j - 1, k);
                count++;
            }
            if (j < ctx.fluidSdf->height() - 1) {
                sum += ctx.fluidSdf->grid(i, j + 1, k);
                count++;
            }
            if (k > 0) {
                sum += ctx.fluidSdf->grid(i, j, k - 1);
                count++;
            }
            if (k < ctx.fluidSdf->depth() - 1) {
                sum += ctx.fluidSdf->grid(i, j, k + 1);
                count++;
            }
            ctx.fluidSdfBuf->grid(i, j, k) = sum / count;
            if (ctx.fluidSdf->grid(i, j, k) < ctx.fluidSdfBuf->grid(i, j, k))
                ctx.fluidSdfBuf->grid(i, j, k) = ctx.fluidSdf->grid(i, j, k);
        });
        std::swap(ctx.fluidSdf, ctx.fluidSdfBuf);
    }
}

void CPUReconstructor::configure(const core::Properties& props) {
    if (auto v = props.tryGet<Real>("particle_radius"))
        config_.particleRadius = *v;
    if (auto v = props.tryGet<int>("smooth_iters"))
        config_.smoothIters = *v;
    if (auto v = props.tryGet<int>("extrapolate_iters"))
        config_.extrapolateIters = *v;
}

} // namespace fluid::cpu
