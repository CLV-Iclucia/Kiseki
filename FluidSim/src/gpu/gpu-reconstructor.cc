// ============================================================================
// src/gpu/gpu-reconstructor.cc
// GPUReconstructor: particle → SDF rebuild + smooth (optional)
// ============================================================================

#include <FluidSim/gpu/gpu-reconstructor.h>
#include <Core/profiler.h>
#include <RHI/rhi.h>

#include <cmath>
#include <iostream>

namespace fluid::gpu {

using namespace ksk::rhi;

GPUReconstructor::GPUReconstructor(Device& device, const GPUGridState& grid)
    : buildHash_(device),
      buildRanges_(device),
      reconstructHashed_(device),
      smooth_(device)
{
    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    const uint32_t nc = static_cast<uint32_t>(nx) *
                        static_cast<uint32_t>(ny) *
                        static_cast<uint32_t>(nz);

    sdfBuf_ = device.createImage({
        .dim = ImageDesc::Dim::D3,
        .width = static_cast<uint32_t>(nx),
        .height = static_cast<uint32_t>(ny),
        .depth = static_cast<uint32_t>(nz),
        .format = Format::R32_Float,
        .usage = ImageDesc::Storage | ImageDesc::Sampled,
        .debugName = "sdfBuf",
    });

    particleCellKeys_ = createDeviceLocalBuffer(
        device, sizeof(uint32_t) * grid.numParticles, "particle-cell-keys");
    particleIndices_ = createDeviceLocalBuffer(
        device, sizeof(uint32_t) * grid.numParticles, "particle-indices");
    cellStart_ = createDeviceLocalBuffer(
        device, sizeof(uint32_t) * nc, "particle-cell-start");
    cellEnd_ = createDeviceLocalBuffer(
        device, sizeof(uint32_t) * nc, "particle-cell-end");

    auto compiler = device.createShaderCompiler();
    if (compiler) {
        sorter_ = std::make_unique<ksk::rpk::Sort>(device, *compiler);
    }

    std::cout << "[GPUReconstructor] Initialized\n";
}

void GPUReconstructor::execute(CommandList& cmd, GPUGridState& grid) {
    SIM_PROFILE_FUNCTION();

    if (!grid.fluidSdfImg || !grid.particlePositions) return;
    if (!buildHash_.valid() || !buildRanges_.valid() ||
        !reconstructHashed_.valid() || !sorter_) return;
    if (grid.numParticles == 0) return;

    int nx = grid.gridSize.x, ny = grid.gridSize.y, nz = grid.gridSize.z;
    uint32_t nc = static_cast<uint32_t>(nx) * ny * nz;
    uint32_t cellGroups = (nc + 255) / 256;

    const float particleRadius = grid.gridSpacing * 1.2f / 1.414f;
    const uint32_t searchRadius = static_cast<uint32_t>(
        std::ceil(particleRadius / grid.gridSpacing)) + 2;
    const uint32_t particleGroups = (grid.numParticles + 255) / 256;

    // Step 1: Build sorted-grid spatial hash.
    {
        SIM_PROFILE_SCOPE("GPUReconstructor/BuildParticleHash");
        BuildParticleHashCS::Params params;
        params.particleCellKeys = particleCellKeys_;
        params.particleIndices = particleIndices_;
        params.positions = grid.particlePositions;
        params.gridSizeX = static_cast<uint32_t>(nx);
        params.gridSizeY = static_cast<uint32_t>(ny);
        params.gridSizeZ = static_cast<uint32_t>(nz);
        params.gridSpacing = grid.gridSpacing;
        params.originX = grid.originX;
        params.originY = grid.originY;
        params.originZ = grid.originZ;
        params.numParticles = grid.numParticles;
        cmd.dispatch(buildHash_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite,
                      BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

    {
        SIM_PROFILE_SCOPE("GPUReconstructor/SortParticleHash");
        sorter_->pairs(cmd, particleCellKeys_, particleIndices_, grid.numParticles);
    }

    // Step 2: Build [start, end) ranges for each occupied grid cell.
    {
        SIM_PROFILE_SCOPE("GPUReconstructor/BuildCellRanges");
        cmd.fillBuffer(cellStart_, 0xffffffffu);
        cmd.fillBuffer(cellEnd_, 0u);
        cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                          BarrierDesc::AccessTransferWrite,
                          BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

        BuildParticleCellRangesCS::Params params;
        params.particleCellKeys = particleCellKeys_;
        params.cellStart = cellStart_;
        params.cellEnd = cellEnd_;
        params.numParticles = grid.numParticles;
        cmd.dispatch(buildRanges_, params, particleGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite,
                      BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

    // Step 3: Reconstruct SDF from nearby particle buckets only.
    {
        SIM_PROFILE_SCOPE("GPUReconstructor/ReconstructSDFHashed");
        ReconstructSdfHashedCS::Params params;
        params.fluidSdf = grid.fluidSdfImg;
        params.positions = grid.particlePositions;
        params.particleIndices = particleIndices_;
        params.cellStart = cellStart_;
        params.cellEnd = cellEnd_;
        params.gridSizeX = static_cast<uint32_t>(nx);
        params.gridSizeY = static_cast<uint32_t>(ny);
        params.gridSizeZ = static_cast<uint32_t>(nz);
        params.gridSpacing = grid.gridSpacing;
        params.originX = grid.originX;
        params.originY = grid.originY;
        params.originZ = grid.originZ;
        params.searchRadius = searchRadius;
        params.particleRadius = particleRadius;
        cmd.dispatch(reconstructHashed_, params, cellGroups, 1, 1);
    }
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessShaderWrite, BarrierDesc::AccessShaderRead);

    // Step 4: Smooth (3 iterations, ping-pong)
    if (smooth_.valid()) {
        SIM_PROFILE_SCOPE("GPUReconstructor/SmoothSDF");
        ImageRef srcSdf = grid.fluidSdfImg;
        ImageRef dstSdf = sdfBuf_;
        for (int i = 0; i < 3; ++i) {
            SIM_PROFILE_SCOPE("GPUReconstructor/SmoothIteration");
            SmoothSdfCS::Params params;
            params.srcSdf   = ImageBinding{srcSdf, grid.sdfSampler};
            params.dstSdf   = dstSdf;
            params.gridSizeX = static_cast<uint32_t>(nx);
            params.gridSizeY = static_cast<uint32_t>(ny);
            params.gridSizeZ = static_cast<uint32_t>(nz);
            cmd.dispatch(smooth_, params, cellGroups, 1, 1);
            cmd.memoryBarrier(BarrierDesc::StageComputeShader,
                              BarrierDesc::StageComputeShader,
                              BarrierDesc::AccessShaderWrite,
                              BarrierDesc::AccessShaderRead |
                                  BarrierDesc::AccessShaderWrite);
            std::swap(srcSdf, dstSdf);
            // Swap references (conceptual — actual swap handled by barrier+reassignment)
        }
        if (srcSdf != grid.fluidSdfImg) {
            std::swap(grid.fluidSdfImg, sdfBuf_);
        }
    }
}

} // namespace fluid::gpu
