// ============================================================================
// include/FluidSim/gpu/gpu-reconstructor.h
// GPUReconstructor: particle → SDF rebuild + smooth (optional)
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-shaders.h>
#include <RPK/sort.h>
#include <RHI/rhi.h>

#include <memory>

namespace fluid::gpu {

class GPUReconstructor {
public:
    explicit GPUReconstructor(ksk::rhi::Device& device,
                              const GPUGridState& grid);

    // Called by Backend at end of step
    void execute(ksk::rhi::CommandList& cmd, GPUGridState& grid);

private:
    // ===== Owned buffers =====
    ksk::rhi::ImageRef sdfBuf_;  // ping-pong buffer for smooth
    ksk::rhi::BufferRef particleCellKeys_;
    ksk::rhi::BufferRef particleIndices_;
    ksk::rhi::BufferRef cellStart_;
    ksk::rhi::BufferRef cellEnd_;

    BuildParticleHashCS buildHash_;
    BuildParticleCellRangesCS buildRanges_;
    ReconstructSdfHashedCS reconstructHashed_;
    SmoothSdfCS smooth_;
    std::unique_ptr<ksk::rpk::Sort> sorter_;
};

} // namespace fluid::gpu
