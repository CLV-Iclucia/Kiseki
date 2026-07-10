// ============================================================================
// include/FluidSim/gpu/gpu-backend.h
// GPUFluidBackend + GPUGridState
// ============================================================================
#pragma once

#include <FluidSim/fluid-backend.h>
#include <FluidSim/fluid-types.h>
#include <FluidSim/gpu/gpu-particle.h>
#include <FluidSim/gpu/gpu-shaders.h>
#include <RHI/rhi.h>
#include <memory>

namespace fluid::gpu {

// ---- Shared GPU grid buffer set (POD, all components share) ----
struct GPUGridState {
    // Staggered velocity faces (SSBO, R32_FLOAT per element)
    ksk::rhi::BufferRef uGrid, vGrid, wGrid;
    // Saved velocity after P2G (before forces/projection), for FLIP delta
    ksk::rhi::BufferRef uGridOld, vGridOld, wGridOld;
    // Ping-pong buffers for extrapolation
    ksk::rhi::BufferRef uGridBuf, vGridBuf, wGridBuf;
    // Validity flags (SSBO, uint32 per element)
    ksk::rhi::BufferRef uValid, vValid, wValid;
    ksk::rhi::BufferRef uValidBuf, vValidBuf, wValidBuf;
    // SDF images (3D, hardware trilinear sampling)
    ksk::rhi::ImageRef  fluidSdfImg;
    ksk::rhi::ImageRef  colliderSdfImg;
    ksk::rhi::SamplerRef sdfSampler;
    // Particles (SOA: separate position and velocity buffers, tightly-packed float3)
    ksk::rhi::BufferRef particlePositions;
    ksk::rhi::BufferRef particleVelocities;
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
    explicit GPUFluidBackend(ksk::rhi::Device& device);
    ~GPUFluidBackend() override;

    // ---- FluidBackend interface ----
    void initialize(const FluidScene& scene) override;
    void step(Real dt) override;
    void readbackParticles(FluidFrame& out) override;
    void updateCollider(const Mesh& mesh) override;
    void updateSolverConfig(const SolverConfig& config) override;

    // ---- GPU-specific queries (NOT in base class) ----
    [[nodiscard]] ksk::rhi::BufferRef particlePositionBuffer()  const { return grid_.particlePositions; }
    [[nodiscard]] ksk::rhi::BufferRef particleVelocityBuffer()  const { return grid_.particleVelocities; }
    [[nodiscard]] ksk::rhi::ImageRef  fluidSdfImage()   const { return grid_.fluidSdfImg; }
    [[nodiscard]] ksk::rhi::ImageRef  colliderSdfImage() const { return grid_.colliderSdfImg; }

private:
    ksk::rhi::Device& device_;

    // ===== Shared state =====
    GPUGridState grid_;

    // ===== Sub-components =====
    std::unique_ptr<GPUAdvector>      advector_;
    std::unique_ptr<GPUProjector>     projector_;
    std::unique_ptr<GPUReconstructor> reconstructor_;

    DirichletCS dirichlet_;
    ExtrapolateCS extrapolate_;
    BodyForceCS bodyForce_;
    ColliderCS collider_;

    // ===== CFL reduction =====
    CflReduceCS cflReduce_;
    CflReduceFinalCS cflReduceFinal_;
    ksk::rhi::BufferRef   cflPartialBuf_;
    ksk::rhi::BufferRef   cflResultBuf_;       // 1 float on device
    ksk::rhi::BufferRef   cflReadbackBuf_;     // 1 float, host-visible
    float                 cachedMaxSpeed_ = 0.0f;

    // ===== Readback (SOA) =====
    ksk::rhi::BufferRef readbackPos_;
    ksk::rhi::BufferRef readbackVel_;

    // ===== Config =====
    SolverConfig config_;

    // ===== Internal methods =====
    void createSharedBuffers(const FluidScene& scene);
    void uploadParticles(const FluidScene& scene);
    void uploadColliderToImage(const Mesh& mesh);
    void computeCFL();
    void substep(ksk::rhi::CommandList& cmd, Real dt);
};

} // namespace fluid::gpu
