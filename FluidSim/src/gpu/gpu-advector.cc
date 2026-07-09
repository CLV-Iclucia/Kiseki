// ============================================================================
// src/gpu/gpu-advector.cc
// GPUAdvector: P2G scatter, normalize, G2P gather, RK3 advect
// ============================================================================

#include <FluidSim/gpu/gpu-advector.h>
#include <RHI/rhi.h>

#include <iostream>

namespace fluid::gpu {

using namespace sim::rhi;

// ============================================================================
// Constructor
// ============================================================================
GPUAdvector::GPUAdvector(Device& device, const GPUGridState& grid)
    : p2g_(device),
      normalize_(device),
      g2p_(device),
      advect_(device)
{
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;

    // Allocate owned buffers (P2G weights, float32 for atomic CAS trick)
    uWeights_ = createDeviceLocalBuffer(
        device, sizeof(float) * static_cast<size_t>(nx + 1) * ny * nz,
        "uWeights");
    vWeights_ = createDeviceLocalBuffer(
        device, sizeof(float) * static_cast<size_t>(nx) * (ny + 1) * nz,
        "vWeights");
    wWeights_ = createDeviceLocalBuffer(
        device, sizeof(float) * static_cast<size_t>(nx) * ny * (nz + 1),
        "wWeights");

    std::cout << "[GPUAdvector] Initialized with "
              << nx << "x" << ny << "x" << nz << " grid\n";
}

// ============================================================================
// scatterP2G: clear → P2G scatter → normalize
// ============================================================================
void GPUAdvector::scatterP2G(CommandList& cmd, GPUGridState& grid, Real dt) {
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t particleGroups = (grid.numParticles + 255) / 256;
    uint32_t totalFaces = static_cast<uint32_t>(
        (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1));
    uint32_t faceGroups = (totalFaces + 255) / 256;

    // Step 1: Clear velocity faces + weights + valid flags
    cmd.fillBuffer(grid.uGrid, 0u);
    cmd.fillBuffer(grid.vGrid, 0u);
    cmd.fillBuffer(grid.wGrid, 0u);
    cmd.fillBuffer(uWeights_, 0u);
    cmd.fillBuffer(vWeights_, 0u);
    cmd.fillBuffer(wWeights_, 0u);
    cmd.fillBuffer(grid.uValid, 0u);
    cmd.fillBuffer(grid.vValid, 0u);
    cmd.fillBuffer(grid.wValid, 0u);
    cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderWrite);

    // Step 2: P2G scatter
    if (p2g_.valid()) {
        P2GScatterCS::Params params;
        params.uGrid       = grid.uGrid;
        params.vGrid       = grid.vGrid;
        params.wGrid       = grid.wGrid;
        params.uWeights    = uWeights_;
        params.vWeights    = vWeights_;
        params.wWeights    = wWeights_;
        params.positions   = grid.particlePositions;
        params.velocities  = grid.particleVelocities;
        params.gridSizeX   = static_cast<uint32_t>(nx);
        params.gridSizeY   = static_cast<uint32_t>(ny);
        params.gridSizeZ   = static_cast<uint32_t>(nz);
        params.gridSpacing = static_cast<float>(grid.gridSpacing);
        params.originX     = grid.originX;
        params.originY     = grid.originY;
        params.originZ     = grid.originZ;
        params.numParticles = grid.numParticles;
        params.dt          = static_cast<float>(dt);
        cmd.dispatch(p2g_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    // Step 3: Normalize
    if (normalize_.valid()) {
        P2GNormalizeCS::Params params;
        params.uGrid     = grid.uGrid;
        params.vGrid     = grid.vGrid;
        params.wGrid     = grid.wGrid;
        params.uWeights  = uWeights_;
        params.vWeights  = vWeights_;
        params.wWeights  = wWeights_;
        params.uValid    = grid.uValid;
        params.vValid    = grid.vValid;
        params.wValid    = grid.wValid;
        params.gridSizeX = static_cast<uint32_t>(nx);
        params.gridSizeY = static_cast<uint32_t>(ny);
        params.gridSizeZ = static_cast<uint32_t>(nz);
        params.maxFaces  = totalFaces;
        cmd.dispatch(normalize_, params, faceGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);
}

// ============================================================================
// gatherAndAdvect: G2P → RK3 advect
// ============================================================================
void GPUAdvector::gatherAndAdvect(CommandList& cmd, GPUGridState& grid, Real dt) {
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t particleGroups = (grid.numParticles + 255) / 256;

    // Step 1: G2P (pure PIC: particle velocity = grid velocity)
    if (g2p_.valid()) {
        G2PGatherCS::Params params;
        params.positions   = grid.particlePositions;
        params.velocities  = grid.particleVelocities;
        params.uGrid       = grid.uGrid;
        params.vGrid       = grid.vGrid;
        params.wGrid       = grid.wGrid;
        params.gridSizeX   = static_cast<uint32_t>(nx);
        params.gridSizeY   = static_cast<uint32_t>(ny);
        params.gridSizeZ   = static_cast<uint32_t>(nz);
        params.gridSpacing = static_cast<float>(grid.gridSpacing);
        params.originX     = grid.originX;
        params.originY     = grid.originY;
        params.originZ     = grid.originZ;
        params.numParticles = grid.numParticles;
        cmd.dispatch(g2p_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);

    // Step 2: Advect (RK2 semi-Lagrangian position update)
    if (advect_.valid()) {
        AdvectCS::Params params;
        params.positions    = grid.particlePositions;
        params.velocities   = grid.particleVelocities;
        params.uGrid        = grid.uGrid;
        params.vGrid        = grid.vGrid;
        params.wGrid        = grid.wGrid;
        params.colliderSdf  = ImageBinding{grid.colliderSdfImg, grid.sdfSampler};
        params.sdfSampler   = grid.sdfSampler;
        params.gridSizeX    = static_cast<uint32_t>(nx);
        params.gridSizeY    = static_cast<uint32_t>(ny);
        params.gridSizeZ    = static_cast<uint32_t>(nz);
        params.gridSpacing  = static_cast<float>(grid.gridSpacing);
        params.originX      = grid.originX;
        params.originY      = grid.originY;
        params.originZ      = grid.originZ;
        params.numParticles = grid.numParticles;
        params.dt           = static_cast<float>(dt);
        cmd.dispatch(advect_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);
}

} // namespace fluid::gpu
