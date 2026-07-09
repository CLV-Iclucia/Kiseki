// ============================================================================
// include/FluidSim/gpu/gpu-advector.h
// GPUAdvector: P2G scatter, normalize, G2P gather, RK3 advect
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-shaders.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUAdvector {
public:
    explicit GPUAdvector(sim::rhi::Device& device,
                         const GPUGridState& grid);

    // Called by Backend in substep
    void scatterP2G(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);
    void gatherAndAdvect(sim::rhi::CommandList& cmd, GPUGridState& grid, Real dt);

private:
    // ===== Owned buffers (P2G weights) =====
    sim::rhi::BufferRef uWeights_, vWeights_, wWeights_;

    P2GScatterCS p2g_;
    P2GNormalizeCS normalize_;
    G2PGatherCS g2p_;
    AdvectCS advect_;
};

} // namespace fluid::gpu
