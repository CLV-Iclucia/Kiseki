// ============================================================================
// include/FluidSim/gpu/gpu-reconstructor.h
// GPUReconstructor: particle → SDF rebuild + smooth (optional)
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUReconstructor {
public:
    explicit GPUReconstructor(sim::rhi::Device& device,
                              sim::rhi::ShaderCompiler& compiler,
                              const GPUGridState& grid);

    // Called by Backend at end of step
    void execute(sim::rhi::CommandList& cmd, GPUGridState& grid);

private:
    // ===== Owned buffers =====
    sim::rhi::ImageRef sdfBuf_;  // ping-pong buffer for smooth

    // ===== Pipelines =====
    sim::rhi::PipelineRef psoReconstruct_;
    sim::rhi::PipelineRef psoSmooth_;
};

} // namespace fluid::gpu
