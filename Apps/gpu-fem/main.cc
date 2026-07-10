//
// SimCraft Example: GPU FEM Backend (C++)
// ========================================
//
// Drives the FEM simulation through the fully GPU-resident FEMBackend: the
// entire implicit-Euler Newton step (elastic + deformable self-contact —
// broad-phase / activation / barrier / CCD + a GPU energy line search) runs on
// the device, via the createFEMBackend("gpu", device, compiler) path. There is
// no CPU assembly and positions stay resident across Newton iterations.
//
// An elastic tet mesh is dropped under gravity onto a triangulated ground
// collider; each step the frame is read back and energies / Newton iterations
// are logged. Optionally dumps surface OBJ frames for offline viewing.
//
// Headless (no window): the GPU device is created for compute only, so this
// runs without GLFW / a display. Requires the Vulkan SDK + dxcompiler.
//
// Usage:
//   gpu-fem [--mesh PATH] [--backend gpu|cpu] [--dt 0.01] [--steps 300]
//           [--raise 1.5] [--ground 0.0] [--dump] [--dump-interval 10]
//

#include <cxxopts.hpp>

#include <RHI/rhi.h>

#include <fem/fem-backend.h>
#include <fem/gpu/gpu-fem-backend.h>
#include <fem/primitives/tet-mesh.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <format>
#include <vector>

using namespace ksk;

// ─── Helper: build a flat triangulated ground quad (grid) ───────────────────
static fem::FEMScene::ColliderDesc::MeshData
makeGroundMesh(double halfExtent, double groundY, int gridN) {
    fem::FEMScene::ColliderDesc::MeshData data;
    const int nSide = gridN + 1;
    data.vertices.reserve(size_t(nSide) * nSide);
    for (int j = 0; j < nSide; ++j)
        for (int i = 0; i < nSide; ++i) {
            double u = double(i) / gridN, v = double(j) / gridN;
            data.vertices.emplace_back((u - 0.5) * 2.0 * halfExtent, groundY,
                                       (v - 0.5) * 2.0 * halfExtent);
        }
    data.triangles.reserve(size_t(gridN) * gridN * 2);
    for (int j = 0; j < gridN; ++j)
        for (int i = 0; i < gridN; ++i) {
            int v00 = j * nSide + i, v10 = j * nSide + (i + 1);
            int v01 = (j + 1) * nSide + i, v11 = (j + 1) * nSide + (i + 1);
            data.triangles.emplace_back(v00, v10, v11);
            data.triangles.emplace_back(v00, v11, v01);
        }
    return data;
}

// ─── Helper: write a surface OBJ for one frame ──────────────────────────────
static void writeObj(const std::string& path,
                     const std::vector<glm::dvec3>& positions,
                     const std::vector<glm::ivec3>& surface) {
    std::ofstream out(path);
    if (!out) { std::cerr << std::format("[gpu-fem] cannot write {}\n", path); return; }
    for (const auto& p : positions)
        out << std::format("v {} {} {}\n", p.x, p.y, p.z);
    for (const auto& t : surface)
        out << std::format("f {} {} {}\n", t.x + 1, t.y + 1, t.z + 1);  // OBJ is 1-based
}

int main(int argc, char** argv) {
    cxxopts::Options options("gpu-fem", "FEM simulation through the GPU FEMBackend");
    options.add_options()
        ("mesh", "Tet mesh (.tobj)", cxxopts::value<std::string>()->default_value(FEM_TETS_DIR "/cube10x10.tobj"))
        ("backend", "Backend: gpu | cpu", cxxopts::value<std::string>()->default_value("gpu"))
        ("dt", "Timestep size", cxxopts::value<double>()->default_value("0.01"))
        ("steps", "Number of steps", cxxopts::value<int>()->default_value("300"))
        ("raise", "Initial height above ground", cxxopts::value<double>()->default_value("1.5"))
        ("ground", "Ground plane y", cxxopts::value<double>()->default_value("0.0"))
        ("dump", "Dump surface OBJ frames", cxxopts::value<bool>()->default_value("false"))
        ("dump-interval", "Steps between OBJ dumps", cxxopts::value<int>()->default_value("10"))
        ("h,help", "Print help");
    auto args = options.parse(argc, argv);
    if (args.count("help")) { std::cout << options.help() << std::endl; return 0; }

    const std::string meshPath = args["mesh"].as<std::string>();
    const std::string backendType = args["backend"].as<std::string>();
    const double dt = args["dt"].as<double>();
    const int maxSteps = args["steps"].as<int>();
    const double raise = args["raise"].as<double>();
    const double groundY = args["ground"].as<double>();
    const bool dump = args["dump"].as<bool>();
    const int dumpInterval = args["dump-interval"].as<int>();

    // ─── 1. Load tet mesh ───────────────────────────────────────────────────
    auto meshOpt = fem::readTetMeshFromTOBJ(meshPath);
    if (!meshOpt) { std::cerr << std::format("[gpu-fem] failed to load mesh: {}\n", meshPath); return 1; }
    const fem::TetMesh& mesh = *meshOpt;

    const auto& verts = mesh.getVertices();
    // Translate the whole rest shape up by `raise` so it starts strain-free
    // above the ground and falls under gravity.
    double minY = 1e300;
    for (const auto& v : verts) minY = std::min(minY, v[1]);
    const double shift = (groundY + raise) - minY;

    // Capture surface triangles (for OBJ dump); indices match vertex ordering.
    std::vector<glm::ivec3> surface(mesh.surfaces.begin(), mesh.surfaces.end());

    // ─── 2. Build FEMScene ──────────────────────────────────────────────────
    fem::FEMScene scene;
    {
        fem::FEMScene::TetMeshDesc desc;
        desc.vertices.reserve(verts.size());
        for (const auto& v : verts)
            desc.vertices.emplace_back(v[0], v[1] + shift, v[2]);
        desc.tets.reserve(mesh.tets.size());
        for (const auto& t : mesh.tets)
            desc.tets.push_back({t[0], t[1], t[2], t[3]});
        desc.constitutiveModel = "stable-neohookean";
        desc.youngModulus = 1e5;
        desc.poissonRatio = 0.4;
        desc.density = 1000.0;
        scene.meshes.push_back(std::move(desc));
    }
    {
        fem::FEMScene::ColliderDesc ground;
        ground.mesh = makeGroundMesh(/*halfExtent=*/5.0, groundY, /*gridN=*/6);
        ground.motionType = "static";
        scene.colliders.push_back(std::move(ground));
    }
    scene.gravity = {0.0, -9.81, 0.0};
    scene.dHat = 1e-2;
    scene.contactStiffness = 1e8;
    scene.convergenceEps = 1e-2;
    scene.pcgMaxIter = 2000;
    scene.pcgTolerance = 1e-6;

    std::cout << std::format("[gpu-fem] mesh: {} verts, {} tets; ground y={}, raise={}\n",
                             verts.size(), mesh.tets.size(), groundY, raise);

    // ─── 3. Create backend ──────────────────────────────────────────────────
    std::unique_ptr<fem::FEMBackend> backend;
    std::unique_ptr<rhi::Device> device;
    std::unique_ptr<rhi::ShaderCompiler> compiler;

    if (backendType == "gpu") {
        device = rhi::Device::create({.backend = rhi::Backend::Vulkan});
        if (!device) { std::cerr << "[gpu-fem] no Vulkan device; try --backend cpu\n"; return 1; }
        compiler = device->createShaderCompiler();
        if (!compiler) { std::cerr << "[gpu-fem] dxcompiler unavailable; try --backend cpu\n"; return 1; }
        backend = fem::createFEMBackend("gpu", *device, *compiler);
        std::cout << "[gpu-fem] using fully GPU-resident backend (device assembly + self-contact + PCG)\n";
    } else {
        backend = fem::createFEMBackend("cpu");  // device-free overload
        std::cout << "[gpu-fem] using CPU backend\n";
    }

    // ─── 4. Initialize + run ────────────────────────────────────────────────
    backend->initialize(scene);

    namespace fs = std::filesystem;
    if (dump) fs::create_directories("gpu-fem-frames");

    fem::FEMFrame frame;
    backend->readback(frame);
    std::cout << std::format("[gpu-fem] t=0  KE={:.4e} PE={:.4e} total={:.4e}\n",
                             frame.kineticEnergy, frame.potentialEnergy, frame.totalEnergy);

    for (int step = 1; step <= maxSteps; ++step) {
        backend->step(dt);
        backend->readback(frame);

        if (step % 25 == 0 || step == 1)
            std::cout << std::format(
                "[gpu-fem] step {:4d} | t={:.3f} | KE={:.4e} | PE={:.4e} | total={:.4e} | newton={}\n",
                step, frame.time, frame.kineticEnergy, frame.potentialEnergy,
                frame.totalEnergy, frame.newtonIters);

        if (dump && step % dumpInterval == 0)
            writeObj(std::format("gpu-fem-frames/frame{:04d}.obj", step), frame.positions, surface);
    }

    std::cout << std::format("[gpu-fem] done: {} steps\n", maxSteps);
    return 0;
}
