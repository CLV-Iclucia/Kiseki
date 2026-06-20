// ============================================================================
// include/FluidSim/gpu/gpu-backend.h
// GPUFluidBackend + GPUGridState
// ============================================================================
#pragma once

#include <FluidSim/fluid-backend.h>
#include <FluidSim/fluid-types.h>
#include <FluidSim/gpu/gpu-particle.h>
#include <RHI/rhi.h>
#include <memory>

namespace fluid::gpu {

// ---- Shared GPU grid buffer set (POD, all components share) ----
struct GPUGridState {
    // Staggered velocity faces (SSBO, R32_FLOAT per element)
    sim::rhi::BufferRef uGrid, vGrid, wGrid;
    // Ping-pong buffers for extrapolation
    sim::rhi::BufferRef uGridBuf, vGridBuf, wGridBuf;
    // Validity flags (SSBO, uint32 per element)
    sim::rhi::BufferRef uValid, vValid, wValid;
    sim::rhi::BufferRef uValidBuf, vValidBuf, wValidBuf;
    // SDF images (3D, hardware trilinear sampling)
    sim::rhi::ImageRef  fluidSdfImg;
    sim::rhi::ImageRef  colliderSdfImg;
    sim::rhi::SamplerRef sdfSampler;
    // Particles (SOA: separate position and velocity buffers, tightly-packed float3)
    sim::rhi::BufferRef particlePositions;
    sim::rhi::BufferRef particleVelocities;
    // Dimensions
    Vec3i    gridSize{64, 64, 64};
    float    gridSpacing{0.015625f};
    float    originX{0.0f}, originY{0.0f}, originZ{0.0f};
    uint32_t numParticles{0};
};

// Forward decls
class GPUAdvector;
class GPUProjector;
class GPUReconstructor;

class GPUFluidBackend : public FluidBackend {
public:
    explicit GPUFluidBackend(sim::rhi::Device& device,
                             sim::rhi::ShaderCompiler& compiler);
    ~GPUFluidBackend() override;

    // ---- FluidBackend interface ----
    void initialize(const FluidScene& scene) override;
    void step(Real dt) override;
    void readbackParticles(FluidFrame& out) override;
    void updateCollider(const Mesh& mesh) override;
    void updateSolverConfig(const SolverConfig& config) override;

    // ---- GPU-specific queries (NOT in base class) ----
    [[nodiscard]] sim::rhi::BufferRef particlePositionBuffer()  const { return grid_.particlePositions; }
    [[nodiscard]] sim::rhi::BufferRef particleVelocityBuffer()  const { return grid_.particleVelocities; }
    [[nodiscard]] sim::rhi::ImageRef  fluidSdfImage()   const { return grid_.fluidSdfImg; }
    [[nodiscard]] sim::rhi::ImageRef  colliderSdfImage() const { return grid_.colliderSdfImg; }

private:
    sim::rhi::Device&          device_;
    sim::rhi::ShaderCompiler&  compiler_;

    // ===== Shared state =====
    GPUGridState grid_;

    // ===== Sub-components =====
    std::unique_ptr<GPUAdvector>      advector_;
    std::unique_ptr<GPUProjector>     projector_;
    std::unique_ptr<GPUReconstructor> reconstructor_;

    // ===== Boundary pipelines =====
    sim::rhi::PipelineRef psoDirichlet_;
    sim::rhi::PipelineRef psoExtrapolate_;
    sim::rhi::PipelineRef psoBodyForce_;
    sim::rhi::PipelineRef psoCollider_;

    // ===== Readback (SOA) =====
    sim::rhi::BufferRef readbackPos_;
    sim::rhi::BufferRef readbackVel_;

    // ===== Config =====
    SolverConfig config_;

    // ===== Internal methods =====
    void createSharedBuffers(const FluidScene& scene);
    void createBoundaryPipelines();
    void uploadParticles(const FluidScene& scene);
    void uploadColliderToImage(const Mesh& mesh);
    void substep(sim::rhi::CommandList& cmd, Real dt);
    [[nodiscard]] Real computeCFL() const;
};

} // namespace fluid::gpu
