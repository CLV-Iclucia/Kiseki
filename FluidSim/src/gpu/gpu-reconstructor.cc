// ============================================================================
// src/gpu/gpu-reconstructor.cc
// GPUReconstructor: particle → SDF rebuild + smooth (optional)
// ============================================================================

#include <FluidSim/gpu/gpu-reconstructor.h>
#include <Core/profiler.h>
#include <RHI/rhi.h>

#include <iostream>

namespace fluid::gpu {

using namespace ksk::rhi;

GPUReconstructor::GPUReconstructor(Device& device, const GPUGridState& grid)
    : reconstruct_(device),
      smooth_(device)
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

    std::cout << "[GPUReconstructor] Initialized\n";
}

void GPUReconstructor::execute(CommandList& cmd, GPUGridState& grid) {
    SIM_PROFILE_FUNCTION();

    if (!grid.fluidSdfImg || !grid.particlePositions) return;
    if (!reconstruct_.valid()) return;
    if (grid.numParticles == 0) return;

    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t nc = static_cast<uint32_t>(nx) * ny * nz;
    uint32_t cellGroups = (nc + 255) / 256;

    // Step 1: Reconstruct SDF
    {
        SIM_PROFILE_SCOPE("GPUReconstructor/ReconstructSDF");
        ReconstructSdfCS::Params params;
        params.fluidSdf       = grid.fluidSdfImg;
        params.positions      = grid.particlePositions;
        params.gridSizeX      = static_cast<uint32_t>(nx);
        params.gridSizeY      = static_cast<uint32_t>(ny);
        params.gridSizeZ      = static_cast<uint32_t>(nz);
        params.gridSpacing    = static_cast<float>(grid.gridSpacing);
        params.originX = 0.0f; params.originY = 0.0f; params.originZ = 0.0f;
        params.numParticles   = grid.numParticles;
        params.particleRadius = grid.gridSpacing * 1.2f / 1.414f;
        cmd.dispatch(reconstruct_, params, cellGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);

    // Step 2: Smooth (3 iterations, ping-pong)
    if (smooth_.valid()) {
        SIM_PROFILE_SCOPE("GPUReconstructor/SmoothSDF");
        for (int i = 0; i < 3; ++i) {
            SIM_PROFILE_SCOPE("GPUReconstructor/SmoothIteration");
            SmoothSdfCS::Params params;
            params.srcSdf   = ImageBinding{grid.fluidSdfImg, grid.sdfSampler};
            params.dstSdf   = sdfBuf_;
            params.gridSizeX = static_cast<uint32_t>(nx);
            params.gridSizeY = static_cast<uint32_t>(ny);
            params.gridSizeZ = static_cast<uint32_t>(nz);
            cmd.dispatch(smooth_, params, cellGroups, 1, 1);
            // Swap references (conceptual — actual swap handled by barrier+reassignment)
        }
    }
}

} // namespace fluid::gpu
