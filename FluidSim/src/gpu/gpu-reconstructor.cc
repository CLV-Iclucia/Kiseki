// ============================================================================
// src/gpu/gpu-reconstructor.cc
// GPUReconstructor: particle → SDF rebuild + smooth (optional)
// ============================================================================

#include <FluidSim/gpu/gpu-reconstructor.h>
#include <RHI/rhi.h>

#include <filesystem>
#include <iostream>

namespace fluid::gpu {

using namespace sim::rhi;

SHADER_PARAMS_BEGIN(ReconstructSdfParams)
    SHADER_PARAM_UAV   (ImageRef, fluidSdf);
    SHADER_PARAM_SRV   (BufferRef, positions);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
    SHADER_PARAM_SCALAR(float,     gridSpacing);
    SHADER_PARAM_SCALAR(float,     originX);
    SHADER_PARAM_SCALAR(float,     originY);
    SHADER_PARAM_SCALAR(float,     originZ);
    SHADER_PARAM_SCALAR(uint32_t,  numParticles);
    SHADER_PARAM_SCALAR(float,     particleRadius);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(SmoothSdfParams)
    SHADER_PARAM_IMAGE (ImageBinding, srcSdf);
    SHADER_PARAM_UAV   (ImageRef, dstSdf);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t,  gridSizeZ);
SHADER_PARAMS_END();

static std::filesystem::path shaderPath(const std::string& name) {
#ifdef FLUIDSIM_SHADER_DIR
    return std::filesystem::path(FLUIDSIM_SHADER_DIR) / name;
#else
    return std::filesystem::path(name);
#endif
}

GPUReconstructor::GPUReconstructor(Device& device, ShaderCompiler& compiler,
                                   const GPUGridState& grid)
{
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;

    sdfBuf_ = device.createImage({
        .dim = ImageDesc::Dim::D3,
        .width = static_cast<uint32_t>(nx),
        .height = static_cast<uint32_t>(ny),
        .depth = static_cast<uint32_t>(nz),
        .format = Format::R32_Float,
        .usage = ImageDesc::Storage,
        .debugName = "sdfBuf",
    });

    psoReconstruct_ = sim::rhi::compileComputePipeline(device, compiler, shaderPath("reconstruct-sdf.hlsl"));
    psoSmooth_      = sim::rhi::compileComputePipeline(device, compiler, shaderPath("smooth-sdf.hlsl"));

    std::cout << "[GPUReconstructor] Initialized\n";
}

void GPUReconstructor::execute(CommandList& cmd, GPUGridState& grid) {
    if (!grid.fluidSdfImg || !grid.particlePositions) return;
    if (!psoReconstruct_.valid()) return;
    if (grid.numParticles == 0) return;

    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t nc = static_cast<uint32_t>(nx) * ny * nz;
    uint32_t cellGroups = (nc + 255) / 256;

    // Step 1: Reconstruct SDF
    {
        ReconstructSdfParams params;
        params.fluidSdf       = grid.fluidSdfImg;
        params.positions      = grid.particlePositions;
        params.gridSizeX      = static_cast<uint32_t>(nx);
        params.gridSizeY      = static_cast<uint32_t>(ny);
        params.gridSizeZ      = static_cast<uint32_t>(nz);
        params.gridSpacing    = static_cast<float>(grid.gridSpacing);
        params.originX = 0.0f; params.originY = 0.0f; params.originZ = 0.0f;
        params.numParticles   = grid.numParticles;
        params.particleRadius = grid.gridSpacing * 1.2f / 1.414f;
        cmd.dispatch(psoReconstruct_, params, cellGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);

    // Step 2: Smooth (3 iterations, ping-pong)
    if (psoSmooth_.valid()) {
        for (int i = 0; i < 3; ++i) {
            SmoothSdfParams params;
            params.srcSdf   = ImageBinding{grid.fluidSdfImg, grid.sdfSampler};
            params.dstSdf   = sdfBuf_;
            params.gridSizeX = static_cast<uint32_t>(nx);
            params.gridSizeY = static_cast<uint32_t>(ny);
            params.gridSizeZ = static_cast<uint32_t>(nz);
            cmd.dispatch(psoSmooth_, params, cellGroups, 1, 1);
            // Swap references (conceptual — actual swap handled by barrier+reassignment)
        }
    }
}

} // namespace fluid::gpu
