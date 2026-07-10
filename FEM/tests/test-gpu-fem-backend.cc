// ============================================================================
// FEM/tests/test-gpu-fem-backend.cc
// Fully GPU-resident GPUFEMBackend: device-resident elastic implicit-Euler step
// with no CPU System. This first cut covers elastic + inertia under gravity
// (no constraints / no contact yet), so we validate it standalone:
//   * a free (unpinned) tet under gravity has its center of mass follow the
//     analytic free fall  Δy ≈ -0.5 g T²  (elastic stores ~no energy for a
//     rigid translation), and
//   * positions/velocities stay finite.
// ============================================================================
#ifdef FEM_GPU_ENABLED

#include <fem/fem-backend.h>
#include <fem/gpu/gpu-fem-backend.h>
#include <RHI/rhi.h>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <cmath>

using namespace ksk;
using namespace ksk::fem;

namespace {

FEMScene makeFreeTetScene() {
    FEMScene scene;
    FEMScene::TetMeshDesc m;
    m.vertices = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    m.tets      = {{0, 1, 2, 3}};
    m.constitutiveModel = "stable-neohookean";
    m.youngModulus = 1e5;
    m.poissonRatio = 0.4;
    m.density      = 1000.0;
    scene.meshes = {m};
    scene.gravity = {0.0, -9.81, 0.0};
    scene.convergenceEps = 1e-3;
    scene.pcgMaxIter     = 2000;
    scene.pcgTolerance   = 1e-10;
    return scene;
}

glm::dvec3 centerOfMass(const FEMFrame& f) {
    glm::dvec3 c(0.0);
    for (auto& p : f.positions) c += p;
    return c / double(f.positions.size());
}

}  // namespace

TEST(GpuFemBackend, FreeFallUnderGravity) {
    auto device = ksk::rhi::Device::create({.backend = ksk::rhi::Backend::Vulkan,
                                            .enableValidation = true});
    if (!device) GTEST_SKIP() << "No Vulkan device";
    auto compiler = device->createShaderCompiler();
    if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

    auto scene = makeFreeTetScene();
    auto gpu = createFEMBackend("gpu", *device, *compiler);
    gpu->initialize(scene);

    FEMFrame f0; gpu->readback(f0);
    glm::dvec3 com0 = centerOfMass(f0);

    const Real dt = 0.005;
    const int  steps = 20;
    for (int s = 0; s < steps; ++s) gpu->step(dt);

    FEMFrame f1; gpu->readback(f1);
    glm::dvec3 com1 = centerOfMass(f1);

    // finiteness
    for (auto& p : f1.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }

    const double T = dt * steps;
    const double expectedDy = -0.5 * 9.81 * T * T;   // implicit Euler ~ analytic for small dt
    double dy = com1.y - com0.y;
    spdlog::info("[test-gpu-fem-backend] COM dy={:.6e} expected~{:.6e}", dy, expectedDy);

    // x/z drift should be ~0; y close to free fall (implicit Euler slightly over-damps,
    // so allow a generous relative band).
    EXPECT_LT(std::abs(com1.x - com0.x), 1e-6);
    EXPECT_LT(std::abs(com1.z - com0.z), 1e-6);
    EXPECT_LT(std::abs(dy - expectedDy), 0.25 * std::abs(expectedDy));
}

#endif // FEM_GPU_ENABLED
