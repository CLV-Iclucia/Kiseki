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

    std::unique_ptr<FaceGridU> u, uBuf;
    std::unique_ptr<FaceGridV> v, vBuf;
    std::unique_ptr<FaceGridW> w, wBuf;

    std::unique_ptr<SDF<3>> colliderSdf;

    std::unique_ptr<Array3D<char>> uValid, vValid, wValid;
    std::unique_ptr<Array3D<char>> uValidBuf, vValidBuf, wValidBuf;

    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;

    std::unique_ptr<SDF<3>> fluidSdf;
    std::unique_ptr<SDF<3>> fluidSdfBuf;
    std::unique_ptr<Array3D<char>> sdfValid, sdfValidBuf;

    Array3D<Real> uw, vw, ww;

    Array3D<Real> pressure;

    std::unique_ptr<Array3D<Real>> density;
    std::unique_ptr<Array3D<Real>> temperature;

    FaceGridU& ug() { return *u; }
    FaceGridV& vg() { return *v; }
    FaceGridW& wg() { return *w; }
    const FaceGridU& ug() const { return *u; }
    const FaceGridV& vg() const { return *v; }
    const FaceGridW& wg() const { return *w; }
};

} // namespace fluid::cpu
