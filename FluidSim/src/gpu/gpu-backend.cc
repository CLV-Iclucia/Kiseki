// ============================================================================
// src/gpu/gpu-backend.cc
// GPUFluidBackend: orchestrate GPU fluid simulation steps
// ============================================================================

#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-advector.h>
#include <FluidSim/gpu/gpu-projector.h>
#include <FluidSim/gpu/gpu-reconstructor.h>

#include <Core/debug.h>
#include <Core/profiler.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>

namespace fluid::gpu {

using namespace ksk::rhi;

// ============================================================================
// Construction / Destruction
// ============================================================================
GPUFluidBackend::GPUFluidBackend(Device& device)
    : device_(device),
      dirichlet_(device),
      extrapolate_(device),
      bodyForce_(device),
      collider_(device),
      cflReduce_(device),
      cflReduceFinal_(device)
{}

GPUFluidBackend::~GPUFluidBackend() = default;

// ============================================================================
// FluidBackend interface
// ============================================================================
void GPUFluidBackend::initialize(const FluidScene& scene) {
    SIM_PROFILE_FUNCTION();

    // 1. Cache config
    config_ = scene.solver;
    grid_.gridSize = scene.domain.resolution;
    grid_.gridSpacing = static_cast<float>(scene.domain.size.x / static_cast<Real>(scene.domain.resolution.x));
    grid_.numParticles = static_cast<uint32_t>(scene.initialFluid.positions.size());

    std::cout << std::format("[GPU] Initializing: {}x{}x{} grid, {} particles, dx={}\n",
        grid_.gridSize.x, grid_.gridSize.y, grid_.gridSize.z,
        grid_.numParticles, grid_.gridSpacing);

    // 2. Allocate shared buffers
    createSharedBuffers(scene);

    // 3. Upload initial particles
    uploadParticles(scene);

    // 4. Upload collider SDF
    if (scene.colliderMesh.has_value()) {
        uploadColliderToImage(*scene.colliderMesh);
    }

    // 5. Create sub-components
    advector_      = std::make_unique<GPUAdvector>(device_, grid_);
    projector_     = std::make_unique<GPUProjector>(device_, grid_, config_);
    reconstructor_ = std::make_unique<GPUReconstructor>(device_, grid_);

    // 6. CFL reduction buffers
    {
        uint32_t cflGroups = (grid_.numParticles + 255) / 256;
        cflPartialBuf_ = createDeviceLocalBuffer(
            device_, sizeof(float) * cflGroups, "cfl-partial");
        cflResultBuf_ = createDeviceLocalBuffer(
            device_, sizeof(float), "cfl-result");
        cflReadbackBuf_ = device_.createBuffer({
            .sizeBytes  = sizeof(float),
            .visibility = BufferDesc::Visibility::Readback,
            .usage      = BufferDesc::TransferDst,
            .debugName  = "cfl-readback",
        });
    }

    // 7. Allocate readback buffers (CPU-visible, SOA)
    readbackPos_ = device_.createBuffer({
        .sizeBytes = particleBufferSize(grid_.numParticles),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-readback-pos",
    });
    readbackVel_ = device_.createBuffer({
        .sizeBytes = particleBufferSize(grid_.numParticles),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-readback-vel",
    });

    std::cout << "[GPU] Initialization complete.\n";
}

void GPUFluidBackend::computeCFL() {
    SIM_PROFILE_SCOPE_COLOR("GPUFluid/ComputeCFL", ksk::core::profiler_colors::kSolver);

    // GPU two-pass reduction: compute max particle speed
    uint32_t numGroups = (grid_.numParticles + 255) / 256;

    auto cmd = device_.beginCommands(QueueType::Compute);

    // Pass 1: per-workgroup max
    if (cflReduce_.valid()) {
        CflReduceCS::Params p;
        p.velocities   = grid_.particleVelocities;
        p.partialMax   = cflPartialBuf_;
        p.numParticles = grid_.numParticles;
        cmd->dispatch(cflReduce_, p, numGroups, 1, 1);
        cmd->memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
    }

    // Pass 2: final reduction
    if (cflReduceFinal_.valid()) {
        CflReduceFinalCS::Params p;
        p.partialMax = cflPartialBuf_;
        p.scalarOut  = cflResultBuf_;
        p.numGroups  = numGroups;
        cmd->dispatch(cflReduceFinal_, p, 1, 1, 1);
        cmd->memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageTransfer);
    }

    // Copy to readback
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(float)}}};
    cmd->copyBuffer(cflResultBuf_, cflReadbackBuf_, region);
    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Compute);

    // Readback
    auto data = cflReadbackBuf_->mapTyped<float>();
    cachedMaxSpeed_ = data[0];
    cflReadbackBuf_->unmap();
}

void GPUFluidBackend::step(Real dt) {
    SIM_PROFILE_FUNCTION();

    // Compute dynamic CFL from particle velocities
    {
        SIM_PROFILE_SCOPE_COLOR("GPUFluid/CFL", ksk::core::profiler_colors::kSolver);
        computeCFL();
    }
    Real cflDt = (cachedMaxSpeed_ > 1e-6)
        ? config_.maxCfl * static_cast<Real>(grid_.gridSpacing) / static_cast<Real>(cachedMaxSpeed_)
        : dt;  // If no motion yet, use full frame dt

    auto cmd = device_.beginCommands(QueueType::Compute);

    Real t = 0.0;
    int substepCnt = 0;
    while (t < dt) {
        Real subDt = std::min(cflDt, dt - t);
        substepCnt++;
        {
            SIM_PROFILE_SCOPE("GPUFluid/Substep");
            substep(*cmd, subDt);
        }
        t += subDt;
    }
    std::cout << std::format("[GPU] {} substeps, cflDt={:.5f}, vmax={:.3f}\n",
                             substepCnt, cflDt, cachedMaxSpeed_);

    // Optional: surface reconstruction
    if (reconstructor_) {
        SIM_PROFILE_SCOPE("GPUFluid/ReconstructSurface");
        reconstructor_->execute(*cmd, grid_);
    }

    {
        SIM_PROFILE_SCOPE("GPUFluid/SubmitStep");
        cmd->end();
        device_.submitAndWait(*cmd, QueueType::Compute);
    }
    SIM_PROFILE_VALUE("GPUFluid/Substeps", substepCnt);
    SIM_PROFILE_VALUE("GPUFluid/CFLDt", cflDt);
    SIM_PROFILE_VALUE("GPUFluid/MaxSpeed", cachedMaxSpeed_);
    SIM_PROFILE_FRAME_MARK();
}

void GPUFluidBackend::readbackParticles(FluidFrame& out) {
    SIM_PROFILE_FUNCTION();

    // Copy GPU particle buffers to readback (SOA)
    size_t bufSize = particleBufferSize(grid_.numParticles);
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bufSize}}};
    cmd->copyBuffer(grid_.particlePositions, readbackPos_, region);
    cmd->copyBuffer(grid_.particleVelocities, readbackVel_, region);
    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Transfer);

    static_assert(ksk::rhi::is_direct_structured_data_v<Vec3f>);
    auto posData = readbackPos_->mapTyped<Vec3f>();
    auto velData = readbackVel_->mapTyped<Vec3f>();
    out.particlePositions.resize(grid_.numParticles);
    out.particleVelocities.resize(grid_.numParticles);
    for (uint32_t i = 0; i < grid_.numParticles; ++i) {
        out.particlePositions[i] = Vec3d(posData[i]);
        out.particleVelocities[i] = Vec3d(velData[i]);
    }
    readbackPos_->unmap();
    readbackVel_->unmap();
}

void GPUFluidBackend::updateCollider(const Mesh& mesh) {
    SIM_PROFILE_FUNCTION();

    uploadColliderToImage(mesh);
}

void GPUFluidBackend::updateSolverConfig(const SolverConfig& config) {
    config_ = config;
    if (projector_) {
        projector_->updateConfig(device_, config);
    }
}

// ============================================================================
// Buffer creation
// ============================================================================
void GPUFluidBackend::createSharedBuffers(const FluidScene& scene) {
    SIM_PROFILE_SCOPE("GPUFluid/CreateSharedBuffers");

    int nx = grid_.gridSize.x, ny = grid_.gridSize.y, nz = grid_.gridSize.z;
    grid_.originX = static_cast<float>(scene.domain.origin.x);
    grid_.originY = static_cast<float>(scene.domain.origin.y);
    grid_.originZ = static_cast<float>(scene.domain.origin.z);

    size_t nU = static_cast<size_t>(nx + 1) * ny * nz;
    size_t nV = static_cast<size_t>(nx) * (ny + 1) * nz;
    size_t nW = static_cast<size_t>(nx) * ny * (nz + 1);

    // Staggered velocity faces (float32) — need TransferSrc for copy to old buffers
    grid_.uGrid = createDeviceLocalBuffer(
        device_, sizeof(float) * nU, "uGrid");
    grid_.vGrid = createDeviceLocalBuffer(
        device_, sizeof(float) * nV, "vGrid");
    grid_.wGrid = createDeviceLocalBuffer(
        device_, sizeof(float) * nW, "wGrid");
    // Old velocity (saved after P2G, before forces/projection) for FLIP delta
    grid_.uGridOld = createDeviceLocalBuffer(
        device_, sizeof(float) * nU, "uGridOld");
    grid_.vGridOld = createDeviceLocalBuffer(
        device_, sizeof(float) * nV, "vGridOld");
    grid_.wGridOld = createDeviceLocalBuffer(
        device_, sizeof(float) * nW, "wGridOld");
    grid_.uGridBuf = createDeviceLocalBuffer(
        device_, sizeof(float) * nU, "uGridBuf");
    grid_.vGridBuf = createDeviceLocalBuffer(
        device_, sizeof(float) * nV, "vGridBuf");
    grid_.wGridBuf = createDeviceLocalBuffer(
        device_, sizeof(float) * nW, "wGridBuf");

    // Valid flags (uint32 per face)
    grid_.uValid = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nU, "uValid");
    grid_.vValid = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nV, "vValid");
    grid_.wValid = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nW, "wValid");
    grid_.uValidBuf = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nU, "uValidBuf");
    grid_.vValidBuf = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nV, "vValidBuf");
    grid_.wValidBuf = createDeviceLocalBuffer(
        device_, sizeof(uint32_t) * nW, "wValidBuf");

    // Particles (SOA: separate position and velocity buffers, tightly-packed float3)
    grid_.particlePositions = createDeviceLocalBuffer(
        device_, particleBufferSize(grid_.numParticles), "particlePositions");
    grid_.particleVelocities = createDeviceLocalBuffer(
        device_, particleBufferSize(grid_.numParticles), "particleVelocities");

    // Fluid SDF (3D Image, R32_Float — hardware trilinear)
    grid_.fluidSdfImg = device_.createImage({
        .dim    = ImageDesc::Dim::D3,
        .width  = static_cast<uint32_t>(nx),
        .height = static_cast<uint32_t>(ny),
        .depth  = static_cast<uint32_t>(nz),
        .format = Format::R32_Float,
        .usage  = ImageDesc::Storage | ImageDesc::Sampled,
        .debugName = "fluidSdfImg",
    });

    // Collider SDF (3D Image, R32_Float)
    grid_.colliderSdfImg = device_.createImage({
        .dim    = ImageDesc::Dim::D3,
        .width  = static_cast<uint32_t>(nx),
        .height = static_cast<uint32_t>(ny),
        .depth  = static_cast<uint32_t>(nz),
        .format = Format::R32_Float,
        .usage  = ImageDesc::Storage | ImageDesc::Sampled,
        .debugName = "colliderSdfImg",
    });

    // Sampler (trilinear + ClampToEdge)
    grid_.sdfSampler = device_.createSampler({
        .magFilter   = SamplerDesc::Filter::Linear,
        .minFilter   = SamplerDesc::Filter::Linear,
        .addressMode = SamplerDesc::AddressMode::ClampToEdge,
    });
}

void GPUFluidBackend::uploadParticles(const FluidScene& scene) {
    SIM_PROFILE_SCOPE("GPUFluid/UploadParticles");

    size_t bufSize = particleBufferSize(grid_.numParticles);

    static_assert(ksk::rhi::is_direct_structured_data_v<Vec3f>);
    std::vector<Vec3f> positions(grid_.numParticles, Vec3f(0.0f));
    std::vector<Vec3f> velocities(grid_.numParticles, Vec3f(0.0f));
    for (uint32_t i = 0; i < grid_.numParticles; ++i) {
        positions[i] = Vec3f(scene.initialFluid.positions[i]);
        if (i < scene.initialFluid.velocities.size()) {
            velocities[i] = Vec3f(scene.initialFluid.velocities[i]);
        }
    }
    const auto positionBytes = ksk::rhi::asStructuredBytes(
        std::span<const Vec3f>(positions));
    const auto velocityBytes = ksk::rhi::asStructuredBytes(
        std::span<const Vec3f>(velocities));
    assert(positionBytes.size() == bufSize);
    assert(velocityBytes.size() == bufSize);

    // Upload positions via staging buffer
    auto stagingPos = device_.createBuffer({
        .sizeBytes = bufSize,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc,
        .debugName = "positions-staging",
    });
    auto* dstPos = stagingPos->map();
    std::memcpy(dstPos, positionBytes.data(), positionBytes.size());
    stagingPos->unmap();

    // Upload velocities via staging buffer
    auto stagingVel = device_.createBuffer({
        .sizeBytes = bufSize,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc,
        .debugName = "velocities-staging",
    });
    auto* dstVel = stagingVel->map();
    std::memcpy(dstVel, velocityBytes.data(), velocityBytes.size());
    stagingVel->unmap();

    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bufSize}}};
    cmd->copyBuffer(stagingPos, grid_.particlePositions, region);
    cmd->copyBuffer(stagingVel, grid_.particleVelocities, region);
    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

void GPUFluidBackend::uploadColliderToImage(const Mesh& mesh) {
    SIM_PROFILE_SCOPE("GPUFluid/UploadColliderSDF");

    // Phase 1: stub — collider SDF upload will be implemented with CPU-side
    // mesh-to-SDF conversion, then upload to 3D image via staging buffer.
    std::cout << "[GPU] Collider SDF upload stub (mesh with "
              << mesh.triangleCount << " triangles)\n";
}

// ============================================================================
// Substep orchestration
// ============================================================================

void GPUFluidBackend::substep(CommandList& cmd, Real dt) {
    SIM_PROFILE_FUNCTION();

    using B = BarrierDesc;
    auto barrier = [&]() { cmd.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };
    auto transferToCompute = [&]() { cmd.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                                                       B::AccessTransferWrite,
                                                       B::AccessShaderRead | B::AccessShaderWrite); };

    int nx = grid_.gridSize.x, ny = grid_.gridSize.y, nz = grid_.gridSize.z;
    uint32_t nU = static_cast<uint32_t>(nx + 1) * ny * nz;
    uint32_t nV = static_cast<uint32_t>(nx) * (ny + 1) * nz;
    uint32_t nW = static_cast<uint32_t>(nx) * ny * (nz + 1);
    float    fdt = static_cast<float>(dt);

    // ---- 1. P2G scatter + normalize ----
    {
        SIM_PROFILE_SCOPE("GPUFluid/P2G");
        advector_->scatterP2G(cmd, grid_, dt);
    }
    // barrier already done inside scatterP2G




    // ---- 2. Body force (gravity on V faces) ----
    if (bodyForce_.valid()) {
        SIM_PROFILE_SCOPE("GPUFluid/BodyForce");
        BodyForceCS::Params p;
        p.vGrid      = grid_.vGrid;
        p.dt         = fdt;
        p.gravityY   = static_cast<float>(config_.gravity.y);
        p.numVFaces  = nV;
        cmd.dispatch(bodyForce_, p, (nV + 255) / 256, 1, 1);
        barrier();
    }

    // ---- 3. Extrapolate velocities (in-place: write result back to grid) ----
    if (extrapolate_.valid()) {
        SIM_PROFILE_SCOPE("GPUFluid/Extrapolate");
        // Pass 1: extrapolate into gridBuf
        {
            ExtrapolateCS::Params p;
            p.gridIn = grid_.uGrid; p.gridOut = grid_.uGridBuf;
            p.validIn = grid_.uValid; p.validOut = grid_.uValidBuf;
            p.faceResX = nx + 1; p.faceResY = ny; p.faceResZ = nz;
            p.numFaces = nU;
            cmd.dispatch(extrapolate_, p, (nU + 255) / 256, 1, 1);
        }
        {
            ExtrapolateCS::Params p;
            p.gridIn = grid_.vGrid; p.gridOut = grid_.vGridBuf;
            p.validIn = grid_.vValid; p.validOut = grid_.vValidBuf;
            p.faceResX = nx; p.faceResY = ny + 1; p.faceResZ = nz;
            p.numFaces = nV;
            cmd.dispatch(extrapolate_, p, (nV + 255) / 256, 1, 1);
        }
        {
            ExtrapolateCS::Params p;
            p.gridIn = grid_.wGrid; p.gridOut = grid_.wGridBuf;
            p.validIn = grid_.wValid; p.validOut = grid_.wValidBuf;
            p.faceResX = nx; p.faceResY = ny; p.faceResZ = nz + 1;
            p.numFaces = nW;
            cmd.dispatch(extrapolate_, p, (nW + 255) / 256, 1, 1);
        }
        barrier();
        // Copy results back: gridBuf → grid (avoid swap to keep buffer handles stable)
        {
            size_t uBytes = sizeof(float) * static_cast<size_t>(nx + 1) * ny * nz;
            size_t vBytes = sizeof(float) * static_cast<size_t>(nx) * (ny + 1) * nz;
            size_t wBytes = sizeof(float) * static_cast<size_t>(nx) * ny * (nz + 1);
            std::array<BufferCopy, 1> uReg{{{0, 0, uBytes}}};
            std::array<BufferCopy, 1> vReg{{{0, 0, vBytes}}};
            std::array<BufferCopy, 1> wReg{{{0, 0, wBytes}}};
            cmd.copyBuffer(grid_.uGridBuf, grid_.uGrid, uReg);
            cmd.copyBuffer(grid_.vGridBuf, grid_.vGrid, vReg);
            cmd.copyBuffer(grid_.wGridBuf, grid_.wGrid, wReg);
            size_t validUBytes = sizeof(uint32_t) * static_cast<size_t>(nx + 1) * ny * nz;
            size_t validVBytes = sizeof(uint32_t) * static_cast<size_t>(nx) * (ny + 1) * nz;
            size_t validWBytes = sizeof(uint32_t) * static_cast<size_t>(nx) * ny * (nz + 1);
            std::array<BufferCopy, 1> vuReg{{{0, 0, validUBytes}}};
            std::array<BufferCopy, 1> vvReg{{{0, 0, validVBytes}}};
            std::array<BufferCopy, 1> vwReg{{{0, 0, validWBytes}}};
            cmd.copyBuffer(grid_.uValidBuf, grid_.uValid, vuReg);
            cmd.copyBuffer(grid_.vValidBuf, grid_.vValid, vvReg);
            cmd.copyBuffer(grid_.wValidBuf, grid_.wValid, vwReg);
        }
        cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                          BarrierDesc::AccessTransferWrite,
                          BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);
    }

    // ---- 4. Pressure solve (build weights → build system → CG → project) ----
    {
        SIM_PROFILE_SCOPE_COLOR("GPUFluid/Pressure", ksk::core::profiler_colors::kSolver);
        projector_->solve(cmd, grid_, dt);
    }
    barrier();

    // ---- 5. G2P gather + advect particles ----
    {
        SIM_PROFILE_SCOPE("GPUFluid/G2PAdvect");
        advector_->gatherAndAdvect(cmd, grid_, dt);
    }
    // final barrier inside gatherAndAdvect
}

} // namespace fluid::gpu
