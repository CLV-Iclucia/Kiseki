// ============================================================================
// src/cpu/cpu-backend.cc
// CPUFluidBackend: 适配现有 CPU solver 组件到新的 FluidBackend 接口
// ============================================================================

#include <FluidSim/cpu/cpu-backend.h>
#include <FluidSim/cpu/util.h>
#include <Core/debug.h>
#include <Core/rand-gen.h>
#include <cassert>
#include <format>
#include <iostream>
#include <span>

namespace fluid::cpu {

// ============================================================================
// FluidBackend 接口实现
// ============================================================================

void CPUFluidBackend::initialize(const FluidScene& scene) {
    // 1. 缓存配置
    config_ = scene.solver;
    domain_ = scene.domain;

    int nx = scene.domain.resolution.x;
    int ny = scene.domain.resolution.y;
    int nz = scene.domain.resolution.z;
    nParticles_ = static_cast<int>(scene.initialFluid.positions.size());
    gridSpacing_ = scene.domain.size.x / static_cast<Real>(nx);

    Vec3d size = scene.domain.size;
    Vec3d orig = scene.domain.origin;

    // 2. 分配网格
    ug_ = std::make_unique<FaceGridU>(scene.domain.resolution, size);
    vg_ = std::make_unique<FaceGridV>(scene.domain.resolution, size);
    wg_ = std::make_unique<FaceGridW>(scene.domain.resolution, size);
    ubuf_ = std::make_unique<FaceGridU>(scene.domain.resolution, size);
    vbuf_ = std::make_unique<FaceGridV>(scene.domain.resolution, size);
    wbuf_ = std::make_unique<FaceGridW>(scene.domain.resolution, size);

    uValid_    = std::make_unique<FlagGrid>(nx + 1, ny, nz);
    vValid_    = std::make_unique<FlagGrid>(nx, ny + 1, nz);
    wValid_    = std::make_unique<FlagGrid>(nx, ny, nz + 1);
    uValidBuf_ = std::make_unique<FlagGrid>(nx + 1, ny, nz);
    vValidBuf_ = std::make_unique<FlagGrid>(nx, ny + 1, nz);
    wValidBuf_ = std::make_unique<FlagGrid>(nx, ny, nz + 1);

    // 3. 分配 P2G 权重与压力数组
    uw_ = RealGrid(nx + 1, ny, nz);
    vw_ = RealGrid(nx, ny + 1, nz);
    ww_ = RealGrid(nx, ny, nz + 1);
    pg_ = RealGrid(nx, ny, nz);

    // 4. 分配 SDF
    fluidSurface_    = std::make_unique<SDF<3>>(scene.domain.resolution,
                                                size, orig);
    fluidSurfaceBuf_ = std::make_unique<SDF<3>>(scene.domain.resolution,
                                                size, orig);
    colliderSdf_     = std::make_unique<SDF<3>>(scene.domain.resolution,
                                                size, orig);
    colliderSdf_->grid.fill(1e9); // 无 collider 时全域在外部（正值），避免 fractionInside 误判
    sdfValid_    = std::make_unique<FlagGrid>(nx, ny, nz);
    sdfValidBuf_ = std::make_unique<FlagGrid>(nx, ny, nz);

    // 5. 初始化粒子
    positions_ = scene.initialFluid.positions;

    // 6. 创建 advector
    Vec3i res(nx, ny, nz);
    Real w = static_cast<Real>(nx) * gridSpacing_;
    Real h = static_cast<Real>(ny) * gridSpacing_;
    Real d = static_cast<Real>(nz) * gridSpacing_;

    switch (config_.advection) {
    case AdvectionMethod::PIC:
        advector_ = std::make_unique<PicAdvector3D>(nParticles_, res, w, h, d);
        break;
    case AdvectionMethod::FLIP:
        advector_ = std::make_unique<FlipAdvectionSolver3D>(nParticles_, res, w, h, d);
        break;
    case AdvectionMethod::APIC:
        // APIC not implemented yet, fallback to FLIP
        advector_ = std::make_unique<FlipAdvectionSolver3D>(nParticles_, res, w, h, d);
        break;
    }

    // 7. 创建 projector (FVM only for now)
    projector_ = std::make_unique<FvmSolver3D>(nx, ny, nz);
    projector_->setCompressedSolver(CompressedSolverMethod::CG,
                                    PreconditionerMethod::ModifiedIncompleteCholesky);

    // 8. 创建 reconstructor
    reconstructor_ = std::make_unique<NaiveReconstructor<Real, 3>>(
        nParticles_, nx, ny, nz, Vector<Real, 3>(w, h, d));

    // 9. 上传 collider SDF (如果有)
    if (scene.colliderMesh.has_value()) {
        updateCollider(*scene.colliderMesh);
    }

    std::cout << "[CPUFluidBackend] Initialized: "
              << nx << "x" << ny << "x" << nz << " grid, "
              << nParticles_ << " particles" << std::endl;
}

void CPUFluidBackend::step(Real dt) {
    Real t = 0;
    int substep_cnt = 0;
    while (t < dt) {
        Real cfl = computeCFL();
        Real subDt = std::min(cfl, dt - t);
        substep_cnt++;
        std::cout << std::format("[CPUFluidBackend] Substep {}, dt = {}",
                                 substep_cnt, subDt) << std::endl;
        substep(subDt);
        t += subDt;
    }
}

void CPUFluidBackend::readbackParticles(FluidFrame& out) {
    out.particlePositions = positions_;
    if (advector_) {
        out.particleVelocities = advector_->velocities();
    } else {
        out.particleVelocities.clear();
    }
}

void CPUFluidBackend::updateCollider(const Mesh& mesh) {
    std::cout << "[CPUFluidBackend] Building collider SDF..." << std::endl;
    spatify::Array3D<int> closest(colliderSdf_->width(),
                                   colliderSdf_->height(),
                                   colliderSdf_->depth());
    spatify::Array3D<int> intersection_cnt(colliderSdf_->width(),
                                            colliderSdf_->height(),
                                            colliderSdf_->depth());
    manifold2SDF(3, closest, intersection_cnt, mesh, colliderSdf_.get());
    std::cout << "[CPUFluidBackend] Done." << std::endl;
}

void CPUFluidBackend::updateSolverConfig(const SolverConfig& config) {
    config_ = config;
    // 如果 advector 或 projector 需要更新，在此处理
    // 目前 solver 配置在构造时已固定，热更新留待后续
}

// ============================================================================
// 内部方法
// ============================================================================

void CPUFluidBackend::clear() {
    uValid_->fill(0);
    vValid_->fill(0);
    wValid_->fill(0);
    uValidBuf_->fill(0);
    vValidBuf_->fill(0);
    wValidBuf_->fill(0);
    vg_->fill(0);
    ug_->fill(0);
    wg_->fill(0);
    ubuf_->fill(0);
    vbuf_->fill(0);
    wbuf_->fill(0);
    uw_.fill(0);
    vw_.fill(0);
    ww_.fill(0);
    pg_.fill(0);
}

void CPUFluidBackend::applyForce(Real dt) const {
    vg_->forEach([this, dt](int i, int j, int k) {
        if (vValid_->at(i, j, k) && j > 0 && j < vg_->height() - 1)
            vg_->at(i, j, k) -= 9.8 * dt;
    });
}

void CPUFluidBackend::applyCollider() const {
    ug_->parallelForEach([&](int i, int j, int k) {
        if (i == 0 || i == ug_->width() - 1) {
            ug_->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos{ug_->indexToCoord(i, j, k)};
        if (colliderSdf_->eval(pos) < 0.0) {
            Vec3d normal{normalize(colliderSdf_->grad(pos))};
            ug_->at(i, j, k) -= ug_->at(i, j, k) * normal.x;
        }
    });
    vg_->parallelForEach([&](int i, int j, int k) {
        if (j == 0 || j == vg_->height() - 1) {
            vg_->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos = vg_->indexToCoord(i, j, k);
        if (colliderSdf_->eval(pos) < 0.0) {
            Vec3d normal{normalize(colliderSdf_->grad(pos))};
            vg_->at(i, j, k) -= vg_->at(i, j, k) * normal.y;
        }
    });
    wg_->parallelForEach([&](int i, int j, int k) {
        if (k == 0 || k == wg_->depth() - 1) {
            wg_->at(i, j, k) = 0.0;
            return;
        }
        Vec3d pos{wg_->indexToCoord(i, j, k)};
        if (colliderSdf_->eval(pos) < 0.0) {
            Vec3d normal{normalize(colliderSdf_->grad(pos))};
            wg_->at(i, j, k) -= wg_->at(i, j, k) * normal.z;
        }
    });
}

void CPUFluidBackend::applyDirichletBoundary() const {
    for (int j = 0; j < ug_->height(); j++) {
        for (int k = 0; k < ug_->depth(); k++) {
            ug_->at(0, j, k) = 0.0;
            uValid_->at(0, j, k) = 1;
            ug_->at(ug_->width() - 1, j, k) = 0.0;
            uValid_->at(ug_->width() - 1, j, k) = 1;
        }
    }
    for (int i = 0; i < vg_->width(); i++) {
        for (int k = 0; k < vg_->depth(); k++) {
            vg_->at(i, 0, k) = 0.0;
            vValid_->at(i, 0, k) = 1;
            vg_->at(i, vg_->height() - 1, k) = 0.0;
            vValid_->at(i, vg_->height() - 1, k) = 1;
        }
    }
    for (int i = 0; i < wg_->width(); i++) {
        for (int j = 0; j < wg_->height(); j++) {
            wg_->at(i, j, 0) = 0.0;
            wValid_->at(i, j, 0) = 1;
            wg_->at(i, j, wg_->depth() - 1) = 0.0;
            wValid_->at(i, j, wg_->depth() - 1) = 1;
        }
    }
}

void CPUFluidBackend::extrapolateFluidSdf(int iters) {
    for (int iter = 0; iter < iters; iter++) {
        sdfValidBuf_->fill(false);
        fluidSurface_->grid.forEach([&](int i, int j, int k) {
            if (sdfValid_->at(i, j, k)) {
                sdfValidBuf_->at(i, j, k) = true;
                return;
            }
            Real sum{0.0};
            int count{0};
            if (i > 0 && sdfValid_->at(i - 1, j, k)) {
                sum += fluidSurface_->grid(i - 1, j, k);
                count++;
            }
            if (i < fluidSurface_->width() - 1 && sdfValid_->at(i + 1, j, k)) {
                sum += fluidSurface_->grid(i + 1, j, k);
                count++;
            }
            if (j > 0 && sdfValid_->at(i, j - 1, k)) {
                sum += fluidSurface_->grid(i, j - 1, k);
                count++;
            }
            if (j < fluidSurface_->height() - 1 && sdfValid_->at(i, j + 1, k)) {
                sum += fluidSurface_->grid(i, j + 1, k);
                count++;
            }
            if (k > 0 && sdfValid_->at(i, j, k - 1)) {
                sum += fluidSurface_->grid(i, j, k - 1);
                count++;
            }
            if (k < fluidSurface_->depth() - 1 && sdfValid_->at(i, j, k + 1)) {
                sum += fluidSurface_->grid(i, j, k + 1);
                count++;
            }
            if (count > 0) {
                fluidSurfaceBuf_->grid(i, j, k) = sum / count;
                sdfValidBuf_->at(i, j, k) = true;
            } else {
                sdfValidBuf_->at(i, j, k) = false;
            }
        });
        std::swap(fluidSurface_, fluidSurfaceBuf_);
        std::swap(sdfValid_, sdfValidBuf_);
    }
}

void CPUFluidBackend::smoothFluidSurface(int iters) {
    for (int iter = 0; iter < iters; iter++) {
        fluidSurfaceBuf_->grid.forEach([&](int i, int j, int k) {
            Real sum = 0;
            int count = 0;
            if (i > 0) {
                sum += fluidSurface_->grid(i - 1, j, k);
                count++;
            }
            if (i < fluidSurface_->width() - 1) {
                sum += fluidSurface_->grid(i + 1, j, k);
                count++;
            }
            if (j > 0) {
                sum += fluidSurface_->grid(i, j - 1, k);
                count++;
            }
            if (j < fluidSurface_->height() - 1) {
                sum += fluidSurface_->grid(i, j + 1, k);
                count++;
            }
            if (k > 0) {
                sum += fluidSurface_->grid(i, j, k - 1);
                count++;
            }
            if (k < fluidSurface_->depth() - 1) {
                sum += fluidSurface_->grid(i, j, k + 1);
                count++;
            }
            fluidSurfaceBuf_->grid(i, j, k) = sum / count;
            if (fluidSurface_->grid(i, j, k) < fluidSurfaceBuf_->grid(i, j, k))
                fluidSurfaceBuf_->grid(i, j, k) = fluidSurface_->grid(i, j, k);
        });
        std::swap(fluidSurface_, fluidSurfaceBuf_);
    }
}

void CPUFluidBackend::substep(Real dt) {
    // 与现有 cpu::FluidSimulator::substep() 流程完全一致
    clear();

    std::cout << "[CPUFluidBackend] Solving advection... ";
    advector_->advect(std::span(positions_), *ug_, *vg_, *wg_, *colliderSdf_, dt);
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Reconstructing surface... ";
    reconstructor_->reconstruct(
        std::span<Vec3d>(positions_),
        0.8 * ug_->gridSpacing().x,
        *fluidSurface_, *sdfValid_);
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Smoothing surface... ";
    extrapolateFluidSdf(10);
    smoothFluidSurface(5);
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Solving P2G... ";
    advector_->solveP2G(std::span(positions_),
                        *ug_, *vg_, *wg_, *colliderSdf_,
                        uw_, vw_, ww_,
                        *uValid_, *vValid_, *wValid_, dt);
    applyDirichletBoundary();
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Extrapolating velocities... ";
    extrapolate(ug_, ubuf_, uValid_, uValidBuf_, 10);
    extrapolate(vg_, vbuf_, vValid_, vValidBuf_, 10);
    extrapolate(wg_, wbuf_, wValid_, wValidBuf_, 10);
    std::cout << "Done." << std::endl;

    applyForce(dt);

    std::cout << "[CPUFluidBackend] Building linear system... ";
    projector_->buildSystem(*ug_, *vg_, *wg_, *fluidSurface_, *colliderSdf_, dt);
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Solving linear system... ";
    if (Real residual = projector_->solvePressure(*fluidSurface_, pg_);
        residual > 1e-4)
        std::cerr << "Warning: projection residual is " << residual << std::endl;
    else
        std::cout << "Projection residual is " << residual << std::endl;
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Doing projection and applying collider... ";
    projector_->project(*ug_, *vg_, *wg_, pg_, *fluidSurface_, *colliderSdf_, dt);
    applyCollider();
    std::cout << "Done." << std::endl;

    std::cout << "[CPUFluidBackend] Solving G2P... ";
    advector_->solveG2P(std::span(positions_), *ug_, *vg_, *wg_, *colliderSdf_, dt);
    std::cout << "Done." << std::endl;
}

Real CPUFluidBackend::computeCFL() const {
    Real h{ug_->gridSpacing().x};
    Real cfl{h / 1e-6};
    ug_->forEach([&cfl, h, this](int x, int y, int z) {
        assert(notNan(ug_->at(x, y, z)));
        if (ug_->at(x, y, z) != 0.0)
            cfl = std::min(cfl, h / abs(ug_->at(x, y, z)));
    });
    vg_->forEach([&cfl, h, this](int x, int y, int z) {
        assert(notNan(vg_->at(x, y, z)));
        if (vg_->at(x, y, z) != 0.0)
            cfl = std::min(cfl, h / abs(vg_->at(x, y, z)));
    });
    wg_->forEach([&cfl, h, this](int x, int y, int z) {
        assert(notNan(wg_->at(x, y, z)));
        if (wg_->at(x, y, z) != 0.0)
            cfl = std::min(cfl, h / abs(wg_->at(x, y, z)));
    });
    return config_.maxCfl * std::max(1e-3, cfl);
}

} // namespace fluid::cpu
