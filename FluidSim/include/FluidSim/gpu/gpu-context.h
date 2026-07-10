// ============================================================================
// include/FluidSim/gpu/gpu-context.h
// GPUFluidContext: GPU 后端的 Context 子类
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <FluidSim/gpu/gpu-particle.h>
#include <RHI/rhi.h>
#include <memory>

namespace fluid::gpu {

class GPUFluidContext : public FluidContext {
public:
    explicit GPUFluidContext(ksk::rhi::Device& device,
                            const FluidDomain& domain,
                            uint32_t numParticles);


    ksk::rhi::BufferRef uGrid, vGrid, wGrid;
    // Ping-pong buffers for extrapolation
    ksk::rhi::BufferRef uGridBuf, vGridBuf, wGridBuf;

    ksk::rhi::BufferRef uValid, vValid, wValid;
    ksk::rhi::BufferRef uValidBuf, vValidBuf, wValidBuf;

    ksk::rhi::ImageRef  colliderSdfImg;
    ksk::rhi::SamplerRef sdfSampler;

    Vec3i gridSize{};
    float gpuGridSpacing{0.015625f};
    float originX{0.0f}, originY{0.0f}, originZ{0.0f};

    ksk::rhi::BufferRef particlePositions;   // [N * 3] floats
    ksk::rhi::BufferRef particleVelocities;  // [N * 3] floats
    uint32_t numParticles{};

    ksk::rhi::ImageRef fluidSdfImg;

    // Readback buffers (SOA)
    ksk::rhi::BufferRef readbackPositions;
    ksk::rhi::BufferRef readbackVelocities;

    // ---- CommandList 管理 ----
    void beginFrame() override;
    void endFrame() override;

    ksk::rhi::Device& device() { return device_; }
    ksk::rhi::CommandList& cmd() { return *activeCmd_; }

private:
    ksk::rhi::Device& device_;
    std::unique_ptr<ksk::rhi::CommandList> activeCmd_;
};

} // namespace fluid::gpu
