//
// SimCraft Example: Fluid Dam Break (C++) — GPU Backend
// =====================================================
//
// A block of fluid collapses under gravity in a box domain.
// Uses the GPU FluidBackend with FLIP advection + PCG pressure solve.
//
// Demonstrates:
//   RHI Device/Compiler → FluidScene → GPUFluidBackend → SimulationApp
//
// Usage:
//   fluid-sim [--dt 0.016] [--steps 500] [--res 32] [--particles 50000] [--no-render]
//

#include <cxxopts.hpp>
#include <FluidSim/fluid-types.h>
#include <FluidSim/fluid-backend.h>
#include <FluidSim/gpu/gpu-backend.h>
#include <Renderer/simulation-app.h>
#include <FluidSim/scene-proxy.h>
#include <RHI/rhi.h>
#include <iostream>
#include <format>
#include <random>

using namespace ksk;

// Generate initial particle positions by rejection sampling in a box region
static fluid::InitialFluid generateDamBreakParticles(
    int count, fluid::FluidDomain domain)
{
    fluid::InitialFluid init;
    init.positions.reserve(count);

    // Fluid block occupies left-bottom quarter of the domain
    double xMin = domain.origin.x + domain.size.x * 0.05;
    double xMax = domain.origin.x + domain.size.x * 0.45;
    double yMin = domain.origin.y + domain.size.y * 0.05;
    double yMax = domain.origin.y + domain.size.y * 0.7;
    double zMin = domain.origin.z + domain.size.z * 0.05;
    double zMax = domain.origin.z + domain.size.z * 0.95;

    std::mt19937 rng(42);
    auto rand = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };

    for (int i = 0; i < count; ++i) {
        init.positions.emplace_back(rand(xMin, xMax), rand(yMin, yMax), rand(zMin, zMax));
    }

    return init;
}

int main(int argc, char** argv) {
    cxxopts::Options options("fluid-sim", "Fluid dam break simulation (GPU)");
    options.add_options()
        ("dt", "Frame timestep", cxxopts::value<double>()->default_value("0.016"))
        ("steps", "Number of frames", cxxopts::value<int>()->default_value("500"))
        ("res", "Grid resolution (uniform)", cxxopts::value<int>()->default_value("64"))
        ("particles", "Number of particles", cxxopts::value<int>()->default_value("50000"))
        ("no-render", "Disable rendering", cxxopts::value<bool>()->default_value("false"))
        ("validation", "Enable GPU validation layers", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print help");
    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    const double dt          = args["dt"].as<double>();
    const int    maxSteps    = args["steps"].as<int>();
    const int    res         = args["res"].as<int>();
    const int    nParticles  = args["particles"].as<int>();
    const bool   noRender    = args["no-render"].as<bool>();
    const bool   validation  = args["validation"].as<bool>();

    // ─── 1. RHI Device + Shader Compiler ───────────────────────────────────────

    auto device = rhi::Device::create({
        .backend          = rhi::Backend::Vulkan,
        .enableValidation = validation,
    });
    if (!device) {
        std::cerr << "[FluidSim] ERROR: Failed to create RHI device (no Vulkan?).\n";
        return 1;
    }

    std::cout << "[FluidSim] RHI device created (Vulkan).\n";

    // ─── 2. Scene Description ──────────────────────────────────────────────────

    fluid::FluidScene scene;
    scene.domain.origin     = {0.0, 0.0, 0.0};
    scene.domain.size       = {1.0, 1.0, 1.0};
    scene.domain.resolution = {res, res, res};

    scene.solver.advection          = fluid::AdvectionMethod::FLIP;
    scene.solver.flipBlend          = 0.97;
    scene.solver.density            = 1000.0;
    scene.solver.gravity            = {0.0, -9.8, 0.0};
    scene.solver.maxCfl             = 5.0;
    scene.solver.pressureMaxIters   = 1000;
    // GPU backend: use simple Jacobi iterations (easier to debug)
    scene.solver.preconditioner     = fluid::PreconditionerMethod::None;

    scene.initialFluid = generateDamBreakParticles(nParticles, scene.domain);

    std::cout << std::format("[FluidSim] Dam break: {}x{}x{} grid, {} particles, dt={}\n",
                             res, res, res, nParticles, dt);

    // ─── 3. GPU Backend ────────────────────────────────────────────────────────

    auto backend = std::make_unique<fluid::gpu::GPUFluidBackend>(*device);
    backend->initialize(scene);

    std::cout << "[FluidSim] GPU backend initialized.\n";

    // ─── 4. Run ────────────────────────────────────────────────────────────────

    renderer::SimulationApp app({
        .windowWidth  = 1280,
        .windowHeight = 720,
        .windowTitle  = "SimCraft - Fluid Dam Break (GPU)",
    });

    app.stepFn = [&](int) {
        backend->step(dt);
    };

    app.buildProxy = [&](int step) {
        return renderer::buildSceneProxyFromFluid(*backend, step);
    };

    app.logInterval = 10;
    app.logFn = [&](int step) {
        std::cout << std::format("[FluidSim] Frame {:4d}\n", step);
    };

    int completed;
    if (noRender) {
        completed = app.runHeadless(maxSteps);
    } else {
        completed = app.run(maxSteps);
    }

    std::cout << std::format("Done. {} frames completed.\n", completed);
    return 0;
}
