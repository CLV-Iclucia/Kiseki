// ============================================================================
// src/gpu/gpu-projector.cc
// GPUProjector: build pressure system, delegate solve, project velocities.
// ============================================================================

#include <FluidSim/gpu/gpu-projector.h>
#include <Core/profiler.h>
#include <RHI/rhi.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>

namespace fluid::gpu {

using namespace ksk::rhi;
// ============================================================================
// Constructor
// ============================================================================
GPUProjector::GPUProjector(Device& device, const GPUGridState& grid,
                           const SolverConfig& config)
    : config_(config),
      buildWeights_(device),
      buildSystem_(device),
      project_(device)
{
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t nc       = static_cast<uint32_t>(nx) * ny * nz;
    uint32_t maxFaces = static_cast<uint32_t>(
        std::max({(nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1)}));

    // Pressure system buffers
    Adiag_ = createDeviceLocalBuffer(device, sizeof(float) * nc, "Adiag");
    for (int d = 0; d < 6; ++d)
        Aneighbour_[d] = createDeviceLocalBuffer(
            device, sizeof(float) * nc, std::format("Aneighbour{}", d));
    rhs_ = createDeviceLocalBuffer(device, sizeof(float) * nc, "rhs");
    active_ = createDeviceLocalBuffer(device, nc, "active"); // uint8 per cell
    pressure_ = createDeviceLocalBuffer(
        device, sizeof(float) * nc, "pressure");

    // Face weights
    faceWeightsU_ = createDeviceLocalBuffer(
        device, sizeof(float) * maxFaces, "faceWeightsU");
    faceWeightsV_ = createDeviceLocalBuffer(
        device, sizeof(float) * maxFaces, "faceWeightsV");
    faceWeightsW_ = createDeviceLocalBuffer(
        device, sizeof(float) * maxFaces, "faceWeightsW");

    // Create solver
    auto ps = makePressureSystem(grid.gridSize);
    solver_ = GPUPressureSolver::create(device, ps, config);

    std::cout << std::format("[GPUProjector] {}x{}x{} grid, solver={}\n",
        nx, ny, nz,
        config.preconditioner == PreconditionerMethod::None ? "Jacobi" : "PCG");
}

// ============================================================================
// Public API
// ============================================================================
void GPUProjector::solve(CommandList& cmd, GPUGridState& grid, Real dt) {
    SIM_PROFILE_FUNCTION();

    {
        SIM_PROFILE_SCOPE_COLOR("GPUProjector/BuildSystem", ksk::core::profiler_colors::kCyan);
        buildWeightsAndSystem(cmd, grid, dt);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    auto ps = makePressureSystem(grid.gridSize);
    {
        SIM_PROFILE_SCOPE_COLOR("GPUProjector/LinearSolve", ksk::core::profiler_colors::kBlue);
        solver_->solve(cmd, ps);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    {
        SIM_PROFILE_SCOPE("GPUProjector/ProjectVelocities");
        projectVelocities(cmd, grid, dt);
    }
}

void GPUProjector::updateConfig(Device& device, const SolverConfig& config)
{
    bool solverTypeChanged =
        (config_.preconditioner != config.preconditioner);

    config_ = config;

    if (solverTypeChanged) {
        // Recreate solver with new type
        auto ps = makePressureSystem({1,1,1}); // sizes don't matter for recreation
        solver_ = GPUPressureSolver::create(device, ps, config);
    } else {
        solver_->updateConfig(config);
    }
}

// ============================================================================
// buildWeightsAndSystem
// ============================================================================
void GPUProjector::buildWeightsAndSystem(CommandList& cmd,
                                         const GPUGridState& grid, Real dt)
{
    SIM_PROFILE_FUNCTION();

    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t nc         = static_cast<uint32_t>(nx) * ny * nz;
    uint32_t totalFaces = static_cast<uint32_t>(
        (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1));
    uint32_t faceGroups = (totalFaces + 255) / 256;
    uint32_t cellGroups = (nc + 255) / 256;

    auto barrier = [&]() { cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader); };

    if (buildWeights_.valid()) {
        SIM_PROFILE_SCOPE("GPUProjector/BuildWeights");
        BuildWeightsCS::Params p;
        p.faceWeightsU = faceWeightsU_; p.faceWeightsV = faceWeightsV_; p.faceWeightsW = faceWeightsW_;
        p.colliderSdf  = ksk::rhi::ImageBinding{grid.colliderSdfImg, grid.sdfSampler};
        p.sdfSampler   = grid.sdfSampler;
        p.gridSizeX = nx; p.gridSizeY = ny; p.gridSizeZ = nz;
        p.gridSpacing = grid.gridSpacing;
        p.originX = grid.originX; p.originY = grid.originY; p.originZ = grid.originZ;
        p.maxFaces = totalFaces;
        cmd.dispatch(buildWeights_, p, faceGroups, 1, 1);
        barrier();
    }

    if (buildSystem_.valid()) {
        SIM_PROFILE_SCOPE("GPUProjector/BuildPressureMatrix");
        BuildSystemCS::Params p;
        p.Adiag = Adiag_;
        p.Aneighbour0 = Aneighbour_[0]; p.Aneighbour1 = Aneighbour_[1];
        p.Aneighbour2 = Aneighbour_[2]; p.Aneighbour3 = Aneighbour_[3];
        p.Aneighbour4 = Aneighbour_[4]; p.Aneighbour5 = Aneighbour_[5];
        p.rhs = rhs_; p.active = active_;
        p.uGrid = grid.uGrid; p.vGrid = grid.vGrid; p.wGrid = grid.wGrid;
        p.faceWeightsU = faceWeightsU_; p.faceWeightsV = faceWeightsV_; p.faceWeightsW = faceWeightsW_;
        p.uValid = grid.uValid; p.vValid = grid.vValid; p.wValid = grid.wValid;
        p.gridSizeX = nx; p.gridSizeY = ny; p.gridSizeZ = nz;
        p.gridSpacing = grid.gridSpacing;
        p.density = static_cast<float>(config_.density);
        p.dt = static_cast<float>(dt);
        p.originX = grid.originX; p.originY = grid.originY; p.originZ = grid.originZ;
        cmd.dispatch(buildSystem_, p, cellGroups, 1, 1);
    }
}

// ============================================================================
// projectVelocities
// ============================================================================
void GPUProjector::projectVelocities(CommandList& cmd,
                                     GPUGridState& grid, Real dt)
{
    SIM_PROFILE_FUNCTION();

    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t totalFaces = static_cast<uint32_t>(
        (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1));
    uint32_t faceGroups = (totalFaces + 255) / 256;

    if (!project_.valid()) return;

    ProjectCS::Params p;
    p.uGrid = grid.uGrid; p.vGrid = grid.vGrid; p.wGrid = grid.wGrid;
    p.pressure = pressure_;
    p.faceWeightsU = faceWeightsU_; p.faceWeightsV = faceWeightsV_; p.faceWeightsW = faceWeightsW_;
    p.gridSizeX = nx; p.gridSizeY = ny; p.gridSizeZ = nz;
    p.gridSpacing = grid.gridSpacing;
    p.density = static_cast<float>(config_.density);
    p.dt = static_cast<float>(dt);
    p.maxFaces = totalFaces;
    cmd.dispatch(project_, p, faceGroups, 1, 1);
}

// ============================================================================
// makePressureSystem (view into owned buffers)
// ============================================================================
PressureSystem GPUProjector::makePressureSystem(Vec3i gridSize) const {
    PressureSystem ps;
    ps.Adiag    = Adiag_;
    for (int d = 0; d < 6; ++d) ps.Aneighbour[d] = Aneighbour_[d];
    ps.rhs      = rhs_;
    ps.active   = active_;
    ps.pressure = pressure_;
    ps.gridSize = gridSize;
    return ps;
}



} // namespace fluid::gpu
