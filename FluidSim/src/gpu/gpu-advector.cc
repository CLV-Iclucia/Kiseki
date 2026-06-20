// ============================================================================
// src/gpu/gpu-advector.cc
// GPUAdvector: P2G scatter, normalize, G2P gather, RK3 advect
// ============================================================================

#include <FluidSim/gpu/gpu-advector.h>
#include <RHI/rhi.h>

#include <iostream>

namespace fluid::gpu {

using namespace sim::rhi;

// ---- SHADER_PARAMS definitions ----

// P2G scatter params (SOA: positions + velocities)
SHADER_PARAMS_BEGIN(P2GScatterParams)
    SHADER_PARAM_UAV   (BufferRef, uGrid);
    SHADER_PARAM_UAV   (BufferRef, vGrid);
    SHADER_PARAM_UAV   (BufferRef, wGrid);
    SHADER_PARAM_UAV   (BufferRef, uWeights);
    SHADER_PARAM_UAV   (BufferRef, vWeights);
    SHADER_PARAM_UAV   (BufferRef, wWeights);
    SHADER_PARAM_SRV   (BufferRef, positions);
    SHADER_PARAM_SRV   (BufferRef, velocities);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
    SHADER_PARAM_SCALAR(float,     gridSpacing);
    SHADER_PARAM_SCALAR(float,     originX);
    SHADER_PARAM_SCALAR(float,     originY);
    SHADER_PARAM_SCALAR(float,     originZ);
    SHADER_PARAM_SCALAR(uint32_t,  numParticles);
    SHADER_PARAM_SCALAR(float,     dt);
SHADER_PARAMS_END();

// P2G normalize params
SHADER_PARAMS_BEGIN(P2GNormalizeParams)
    SHADER_PARAM_UAV   (BufferRef, uGrid);
    SHADER_PARAM_UAV   (BufferRef, vGrid);
    SHADER_PARAM_UAV   (BufferRef, wGrid);
    SHADER_PARAM_SRV   (BufferRef, uWeights);
    SHADER_PARAM_SRV   (BufferRef, vWeights);
    SHADER_PARAM_SRV   (BufferRef, wWeights);
    SHADER_PARAM_UAV   (BufferRef, uValid);
    SHADER_PARAM_UAV   (BufferRef, vValid);
    SHADER_PARAM_UAV   (BufferRef, wValid);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
    SHADER_PARAM_SCALAR(uint32_t,  maxFaces);
SHADER_PARAMS_END();

// G2P gather params (SOA: positions + velocities as UAV for read+write)
SHADER_PARAMS_BEGIN(G2PGatherParams)
    SHADER_PARAM_SRV   (BufferRef, positions);
    SHADER_PARAM_UAV   (BufferRef, velocities);
    SHADER_PARAM_SRV   (BufferRef, uGrid);
    SHADER_PARAM_SRV   (BufferRef, vGrid);
    SHADER_PARAM_SRV   (BufferRef, wGrid);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
    SHADER_PARAM_SCALAR(float,     gridSpacing);
    SHADER_PARAM_SCALAR(float,     originX);
    SHADER_PARAM_SCALAR(float,     originY);
    SHADER_PARAM_SCALAR(float,     originZ);
    SHADER_PARAM_SCALAR(uint32_t,  numParticles);
    SHADER_PARAM_SCALAR(float,     flipBlend);
SHADER_PARAMS_END();

// Advect params (RK2 semi-Lagrangian with grid velocity sampling, SOA)
SHADER_PARAMS_BEGIN(AdvectParams)
    SHADER_PARAM_UAV   (BufferRef, positions);
    SHADER_PARAM_UAV   (BufferRef, velocities);
    SHADER_PARAM_SRV   (BufferRef, uGrid);
    SHADER_PARAM_SRV   (BufferRef, vGrid);
    SHADER_PARAM_SRV   (BufferRef, wGrid);
    SHADER_PARAM_IMAGE (ImageBinding, colliderSdf);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
    SHADER_PARAM_SCALAR(float,     gridSpacing);
    SHADER_PARAM_SCALAR(float,     originX);
    SHADER_PARAM_SCALAR(float,     originY);
    SHADER_PARAM_SCALAR(float,     originZ);
    SHADER_PARAM_SCALAR(uint32_t,  numParticles);
    SHADER_PARAM_SCALAR(float,     dt);
SHADER_PARAMS_END();

// ---- Shader path helper ----
static std::filesystem::path shaderPath(const std::string& name) {
#ifdef FLUIDSIM_SHADER_DIR
    return std::filesystem::path(FLUIDSIM_SHADER_DIR) / name;
#else
    return std::filesystem::path(name);
#endif
}

// ============================================================================
// Constructor
// ============================================================================
GPUAdvector::GPUAdvector(Device& device, ShaderCompiler& compiler,
                         const GPUGridState& grid)
{
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;

    // Allocate owned buffers (P2G weights, float32 for atomic CAS trick)
    uWeights_ = device.createBuffer({
        .sizeBytes = sizeof(float) * static_cast<size_t>(nx + 1) * ny * nz,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName = "uWeights",
    });
    vWeights_ = device.createBuffer({
        .sizeBytes = sizeof(float) * static_cast<size_t>(nx) * (ny + 1) * nz,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName = "vWeights",
    });
    wWeights_ = device.createBuffer({
        .sizeBytes = sizeof(float) * static_cast<size_t>(nx) * ny * (nz + 1),
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName = "wWeights",
    });

    // Create pipelines
    psoP2G_       = sim::rhi::compileComputePipeline(device, compiler, shaderPath("p2g-scatter.hlsl"));
    psoNormalize_ = sim::rhi::compileComputePipeline(device, compiler, shaderPath("p2g-normalize.hlsl"));
    psoG2P_       = sim::rhi::compileComputePipeline(device, compiler, shaderPath("g2p-gather.hlsl"));
    psoAdvect_    = sim::rhi::compileComputePipeline(device, compiler, shaderPath("advect.hlsl"));

    std::cout << "[GPUAdvector] Initialized with "
              << nx << "x" << ny << "x" << nz << " grid\n";
}

// ============================================================================
// scatterP2G: clear → P2G scatter → normalize
// ============================================================================
void GPUAdvector::scatterP2G(CommandList& cmd, GPUGridState& grid, Real dt) {
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t particleGroups = (grid.numParticles + 255) / 256;
    uint32_t maxFaces = static_cast<uint32_t>(
        std::max({(nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1)}));
    uint32_t faceGroups = (maxFaces + 255) / 256;

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
    if (psoP2G_.valid()) {
        P2GScatterParams params;
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
        cmd.dispatch(psoP2G_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    // Step 3: Normalize
    if (psoNormalize_.valid()) {
        P2GNormalizeParams params;
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
        params.maxFaces  = maxFaces;
        cmd.dispatch(psoNormalize_, params, faceGroups, 1, 1);
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

    // Step 1: G2P
    if (psoG2P_.valid()) {
        G2PGatherParams params;
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
        params.flipBlend    = 0.97f;  // from config
        cmd.dispatch(psoG2P_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);

    // Step 2: Advect (RK2 semi-Lagrangian position update)
    if (psoAdvect_.valid()) {
        AdvectParams params;
        params.positions    = grid.particlePositions;
        params.velocities   = grid.particleVelocities;
        params.uGrid        = grid.uGrid;
        params.vGrid        = grid.vGrid;
        params.wGrid        = grid.wGrid;
        params.colliderSdf  = ImageBinding{grid.colliderSdfImg, grid.sdfSampler};
        params.gridSizeX    = static_cast<uint32_t>(nx);
        params.gridSizeY    = static_cast<uint32_t>(ny);
        params.gridSizeZ    = static_cast<uint32_t>(nz);
        params.gridSpacing  = static_cast<float>(grid.gridSpacing);
        params.originX      = grid.originX;
        params.originY      = grid.originY;
        params.originZ      = grid.originZ;
        params.numParticles = grid.numParticles;
        params.dt           = static_cast<float>(dt);
        cmd.dispatch(psoAdvect_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);
}

} // namespace fluid::gpu
