// ============================================================================
// include/FluidSim/gpu/gpu-advector.h
// GPUAdvector: P2G scatter, normalize, G2P gather, RK3 advect
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUAdvector {
public:
    explicit GPUAdvector(sim::rhi::Device& device,
                         sim::rhi::ShaderCompiler& compiler,
                         const GPUGridState& grid);

    // Called by Backend in substep
    void scatterP2G(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);
    void gatherAndAdvect(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);

private:
    // ===== Owned buffers (P2G weights) =====
    sim::rhi::BufferRef uWeights_, vWeights_, wWeights_;

    // ===== Pipelines =====
    sim::rhi::PipelineRef psoP2G_;
    sim::rhi::PipelineRef psoNormalize_;
    sim::rhi::PipelineRef psoG2P_;
    sim::rhi::PipelineRef psoAdvect_;
};

} // namespace fluid::gpu
