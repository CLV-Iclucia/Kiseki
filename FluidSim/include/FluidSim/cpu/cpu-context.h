// ============================================================================
// include/FluidSim/cpu/cpu-context.h
// CPUFluidContext: CPU 后端的 Context 子类，持有所有 CPU 端数据
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <FluidSim/cpu/sdf.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <vector>
#include <memory>

namespace fluid::cpu {

using spatify::Array3D;
using spatify::FaceCentredGrid;

class CPUFluidContext : public FluidContext {
public:
    explicit CPUFluidContext(const FluidDomain& dom);

    using FaceGridU = FaceCentredGrid<Real, Real, 3, 0>;
    using FaceGridV = FaceCentredGrid<Real, Real, 3, 1>;
    using FaceGridW = FaceCentredGrid<Real, Real, 3, 2>;

    // ============ 所有模拟都有 ============

    // 速度场（MAC staggered）
    std::unique_ptr<FaceGridU> u, uBuf;
    std::unique_ptr<FaceGridV> v, vBuf;
    std::unique_ptr<FaceGridW> w, wBuf;

    // 碰撞体 SDF
    std::unique_ptr<SDF<3>> colliderSdf;

    // 有效性标记（外推用）
    std::unique_ptr<Array3D<char>> uValid, vValid, wValid;
    std::unique_ptr<Array3D<char>> uValidBuf, vValidBuf, wValidBuf;

    // ============ 自由液面专用（可选）============

    // 粒子
    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;

    // 流体 SDF
    std::unique_ptr<SDF<3>> fluidSdf;
    std::unique_ptr<SDF<3>> fluidSdfBuf;
    std::unique_ptr<Array3D<char>> sdfValid, sdfValidBuf;

    // P2G 权重
    Array3D<Real> uw, vw, ww;

    // 压力
    Array3D<Real> pressure;

    // ============ 烟雾专用（可选）============

    std::unique_ptr<Array3D<Real>> density;
    std::unique_ptr<Array3D<Real>> temperature;

    // ============ 便捷访问 ============

    FaceGridU& ug() { return *u; }
    FaceGridV& vg() { return *v; }
    FaceGridW& wg() { return *w; }
    const FaceGridU& ug() const { return *u; }
    const FaceGridV& vg() const { return *v; }
    const FaceGridW& wg() const { return *w; }
};

} // namespace fluid::cpu
