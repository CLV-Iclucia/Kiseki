// ============================================================================
// include/FluidSim/cpu/cpu-backend.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-backend.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <FluidSim/cpu/sdf.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <memory>
#include <vector>

namespace fluid::cpu {

class CPUFluidBackend : public FluidBackend {
public:
    CPUFluidBackend() = default;

    // ---- FluidBackend 接口 ----
    void initialize(const FluidScene& scene) override;
    void step(Real dt) override;
    void readbackParticles(FluidFrame& out) override;
    void updateCollider(const Mesh& mesh) override;
    void updateSolverConfig(const SolverConfig& config) override;

    // ---- CPU 专用查询 (非基类方法) ----
    [[nodiscard]] const std::vector<Vec3d>& positions() const { return positions_; }
    [[nodiscard]] const SDF<3>& fluidSurface() const { return *fluidSurface_; }
    [[nodiscard]] const SDF<3>& colliderSurface() const { return *colliderSdf_; }

private:
    // ===== 类型别名 =====
    using RealGrid  = spatify::Array3D<Real>;
    using FlagGrid  = spatify::Array3D<char>;
    using FaceGridU = spatify::FaceCentredGrid<Real, Real, 3, 0>;
    using FaceGridV = spatify::FaceCentredGrid<Real, Real, 3, 1>;
    using FaceGridW = spatify::FaceCentredGrid<Real, Real, 3, 2>;

    // ===== 网格状态 (与现有 FluidSimulator 一致) =====
    std::unique_ptr<FaceGridU> ug_, ubuf_;
    std::unique_ptr<FaceGridV> vg_, vbuf_;
    std::unique_ptr<FaceGridW> wg_, wbuf_;
    RealGrid uw_, vw_, ww_;      // P2G 权重
    RealGrid pg_;                 // 压力
    std::unique_ptr<FlagGrid> uValid_, vValid_, wValid_;
    std::unique_ptr<FlagGrid> uValidBuf_, vValidBuf_, wValidBuf_;

    // SDF
    std::unique_ptr<SDF<3>> fluidSurface_, fluidSurfaceBuf_, colliderSdf_;
    std::unique_ptr<FlagGrid> sdfValid_, sdfValidBuf_;

    // 粒子
    std::vector<Vec3d> positions_;

    // ===== 求解器组件 (复用现有代码) =====
    std::unique_ptr<HybridAdvectionSolver3D> advector_;
    std::unique_ptr<ProjectionSolver3D> projector_;
    std::unique_ptr<ParticleSystemReconstructor<Real, 3>> reconstructor_;

    // ===== 配置 =====
    SolverConfig config_;
    FluidDomain  domain_;
    int nParticles_{0};
    Real gridSpacing_{0.0};

    // ===== 内部方法 =====
    void clear();
    void applyForce(Real dt) const;
    void applyCollider() const;
    void applyDirichletBoundary() const;
    void extrapolateFluidSdf(int iters);
    void smoothFluidSurface(int iters);

    template<typename GridType>
    void extrapolate(std::unique_ptr<GridType>& g,
                     std::unique_ptr<GridType>& gbuf,
                     std::unique_ptr<FlagGrid>& valid,
                     std::unique_ptr<FlagGrid>& validBuf, int iters);

    void substep(Real dt);
    [[nodiscard]] Real computeCFL() const;
};

// ===== 模板方法实现 (必须在头文件中) =====
template<typename GridType>
void CPUFluidBackend::extrapolate(
    std::unique_ptr<GridType>& g,
    std::unique_ptr<GridType>& gbuf,
    std::unique_ptr<FlagGrid>& valid,
    std::unique_ptr<FlagGrid>& validBuf, int iters)
{
    for (int iter = 0; iter < iters; iter++) {
        validBuf->fill(false);
        g->parallelForEach([&](int i, int j, int k) {
            if (valid->at(i, j, k)) {
                validBuf->at(i, j, k) = true;
                return;
            }
            Real sum{0.0};
            int count{0};
            if (i > 0 && valid->at(i - 1, j, k)) {
                sum += g->at(i - 1, j, k);
                count++;
            }
            if (i < g->width() - 1 && valid->at(i + 1, j, k)) {
                sum += g->at(i + 1, j, k);
                count++;
            }
            if (j > 0 && valid->at(i, j - 1, k)) {
                sum += g->at(i, j - 1, k);
                count++;
            }
            if (j < g->height() - 1 && valid->at(i, j + 1, k)) {
                sum += g->at(i, j + 1, k);
                count++;
            }
            if (k > 0 && valid->at(i, j, k - 1)) {
                sum += g->at(i, j, k - 1);
                count++;
            }
            if (k < g->depth() - 1 && valid->at(i, j, k + 1)) {
                sum += g->at(i, j, k + 1);
                count++;
            }
            if (count > 0) {
                gbuf->at(i, j, k) = sum / count;
                validBuf->at(i, j, k) = true;
            } else {
                gbuf->at(i, j, k) = 0.0;
                validBuf->at(i, j, k) = false;
            }
        });
        std::swap(g, gbuf);
        std::swap(valid, validBuf);
    }
}

} // namespace fluid::cpu
