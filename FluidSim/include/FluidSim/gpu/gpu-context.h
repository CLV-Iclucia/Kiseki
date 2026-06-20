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
    explicit GPUFluidContext(sim::rhi::Device& device,
                            sim::rhi::ShaderCompiler& compiler,
                            const FluidDomain& domain,
                            uint32_t numParticles);

    // ============ 所有模拟都有 ============

    // 速度场 (SSBO)
    sim::rhi::BufferRef uGrid, vGrid, wGrid;
    // Ping-pong buffers for extrapolation
    sim::rhi::BufferRef uGridBuf, vGridBuf, wGridBuf;

    // 有效性标记 (SSBO)
    sim::rhi::BufferRef uValid, vValid, wValid;
    sim::rhi::BufferRef uValidBuf, vValidBuf, wValidBuf;

    // 碰撞体 SDF (Image 3D)
    sim::rhi::ImageRef  colliderSdfImg;
    sim::rhi::SamplerRef sdfSampler;

    // 维度
    Vec3i gridSize{};
    float gpuGridSpacing{0.015625f};
    float originX{0.0f}, originY{0.0f}, originZ{0.0f};

    // ============ 自由液面专用 ============

    // 粒子 SOA (SSBO: tightly-packed float3)
    sim::rhi::BufferRef particlePositions;   // [N * 3] floats
    sim::rhi::BufferRef particleVelocities;  // [N * 3] floats
    uint32_t numParticles{};

    // 流体 SDF (Image 3D)
    sim::rhi::ImageRef fluidSdfImg;

    // Readback buffers (SOA)
    sim::rhi::BufferRef readbackPositions;
    sim::rhi::BufferRef readbackVelocities;

    // ---- CommandList 管理 ----
    void beginFrame() override;
    void endFrame() override;

    sim::rhi::Device& device() { return device_; }
    sim::rhi::ShaderCompiler& compiler() { return compiler_; }
    sim::rhi::CommandList& cmd() { return *activeCmd_; }

private:
    sim::rhi::Device& device_;
    sim::rhi::ShaderCompiler& compiler_;
    std::unique_ptr<sim::rhi::CommandList> activeCmd_;
};

} // namespace fluid::gpu
