// ============================================================================
// src/cpu/cpu-context.cc
// ============================================================================

#include <FluidSim/cpu/cpu-context.h>

namespace fluid::cpu {

CPUFluidContext::CPUFluidContext(const FluidDomain& dom) {
    // 设置基类成员
    domain = dom;
    gridSpacing = dom.size.x / static_cast<Real>(dom.resolution.x);

    Vec3i res = dom.resolution;
    Vec3d size = dom.size;
    Vec3d orig = dom.origin;

    // 分配速度场
    u    = std::make_unique<FaceGridU>(res, size);
    v    = std::make_unique<FaceGridV>(res, size);
    w    = std::make_unique<FaceGridW>(res, size);
    uBuf = std::make_unique<FaceGridU>(res, size);
    vBuf = std::make_unique<FaceGridV>(res, size);
    wBuf = std::make_unique<FaceGridW>(res, size);

    // 分配有效性标记
    uValid    = std::make_unique<Array3D<char>>(res.x + 1, res.y, res.z);
    vValid    = std::make_unique<Array3D<char>>(res.x, res.y + 1, res.z);
    wValid    = std::make_unique<Array3D<char>>(res.x, res.y, res.z + 1);
    uValidBuf = std::make_unique<Array3D<char>>(res.x + 1, res.y, res.z);
    vValidBuf = std::make_unique<Array3D<char>>(res.x, res.y + 1, res.z);
    wValidBuf = std::make_unique<Array3D<char>>(res.x, res.y, res.z + 1);

    // 分配碰撞体 SDF
    colliderSdf = std::make_unique<SDF<3>>(res, size, orig);

    // 分配自由液面 SDF
    fluidSdf    = std::make_unique<SDF<3>>(res, size, orig);
    fluidSdfBuf = std::make_unique<SDF<3>>(res, size, orig);
    sdfValid    = std::make_unique<Array3D<char>>(res.x, res.y, res.z);
    sdfValidBuf = std::make_unique<Array3D<char>>(res.x, res.y, res.z);

    // 分配 P2G 权重
    uw = Array3D<Real>(res.x + 1, res.y, res.z);
    vw = Array3D<Real>(res.x, res.y + 1, res.z);
    ww = Array3D<Real>(res.x, res.y, res.z + 1);

    // 分配压力
    pressure = Array3D<Real>(res.x, res.y, res.z);
}

} // namespace fluid::cpu
