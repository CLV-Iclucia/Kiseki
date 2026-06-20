// ============================================================================
// src/gpu/gpu-backend.cc
// GPUFluidBackend: orchestrate GPU fluid simulation steps
// ============================================================================

#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-advector.h>
#include <FluidSim/gpu/gpu-projector.h>
#include <FluidSim/gpu/gpu-reconstructor.h>

#include <Core/debug.h>
#include <cassert>
#include <format>
#include <iostream>
#include <filesystem>

namespace fluid::gpu {

using namespace sim::rhi;
namespace fs = std::filesystem;

// ============================================================================
// SHADER_PARAMS for boundary shaders (owned by GPUFluidBackend)
// ============================================================================

SHADER_PARAMS_BEGIN(BodyForceParams)
    SHADER_PARAM_UAV   (BufferRef, vGrid);
    SHADER_PARAM_SCALAR(float,     dt);
    SHADER_PARAM_SCALAR(float,     gravityY);
    SHADER_PARAM_SCALAR(uint32_t,  numVFaces);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ExtrapolateParams)
    SHADER_PARAM_SRV   (BufferRef, gridIn);
    SHADER_PARAM_UAV   (BufferRef, gridOut);
    SHADER_PARAM_SRV   (BufferRef, validIn);
    SHADER_PARAM_UAV   (BufferRef, validOut);
    SHADER_PARAM_SCALAR(uint32_t,  faceResX);
    SHADER_PARAM_SCALAR(uint32_t,  faceResY);
    SHADER_PARAM_SCALAR(uint32_t,  faceResZ);
    SHADER_PARAM_SCALAR(uint32_t,  numFaces);
SHADER_PARAMS_END();

// ============================================================================
// Shader path helper
// ============================================================================
static fs::path shaderPath(const std::string& name) {
#ifdef FLUIDSIM_SHADER_DIR
    return fs::path(FLUIDSIM_SHADER_DIR) / name;
#else
    return fs::path(name);
#endif
}

static fs::path shaderDir() {
#ifdef FLUIDSIM_SHADER_DIR
    return fs::path(FLUIDSIM_SHADER_DIR);
#else
    return fs::path{};
#endif
}

// ============================================================================
// Construction / Destruction
// ============================================================================
GPUFluidBackend::GPUFluidBackend(Device& device, ShaderCompiler& compiler)
    : device_(device), compiler_(compiler) {}

GPUFluidBackend::~GPUFluidBackend() = default;

// ============================================================================
// FluidBackend interface
// ============================================================================
void GPUFluidBackend::initialize(const FluidScene& scene) {
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
    advector_      = std::make_unique<GPUAdvector>(device_, compiler_, grid_);
    projector_     = std::make_unique<GPUProjector>(device_, compiler_, grid_, config_, shaderDir());
    reconstructor_ = std::make_unique<GPUReconstructor>(device_, compiler_, grid_);

    // 6. Create boundary pipelines
    createBoundaryPipelines();

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

void GPUFluidBackend::step(Real dt) {
    auto cmd = device_.beginCommands(QueueType::Compute);

    Real t = 0.0;
    int substepCnt = 0;
    while (t < dt) {
        Real subDt = std::min(computeCFL(), dt - t);
        substepCnt++;
        std::cout << std::format("[GPU] Substep {}, dt={}\n", substepCnt, subDt);
        substep(*cmd, subDt);
        t += subDt;
    }

    // Optional: surface reconstruction
    if (reconstructor_) {
        reconstructor_->execute(*cmd, grid_);
    }

    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Compute);
}

void GPUFluidBackend::readbackParticles(FluidFrame& out) {
    // Copy GPU particle buffers to readback (SOA)
    size_t bufSize = particleBufferSize(grid_.numParticles);
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bufSize}}};
    cmd->copyBuffer(grid_.particlePositions, readbackPos_, region);
    cmd->copyBuffer(grid_.particleVelocities, readbackVel_, region);
    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Transfer);

    // Map + copy (SOA: tightly-packed float3)
    auto posData = readbackPos_->mapTyped<float>();
    auto velData = readbackVel_->mapTyped<float>();
    out.particlePositions.resize(grid_.numParticles);
    out.particleVelocities.resize(grid_.numParticles);
    for (uint32_t i = 0; i < grid_.numParticles; ++i) {
        uint32_t base = i * 3;
        out.particlePositions[i]  = Vec3d(posData[base], posData[base+1], posData[base+2]);
        out.particleVelocities[i] = Vec3d(velData[base], velData[base+1], velData[base+2]);
    }
    readbackPos_->unmap();
    readbackVel_->unmap();
}

void GPUFluidBackend::updateCollider(const Mesh& mesh) {
    uploadColliderToImage(mesh);
}

void GPUFluidBackend::updateSolverConfig(const SolverConfig& config) {
    config_ = config;
    if (projector_) {
        projector_->updateConfig(device_, compiler_, config, shaderDir());
    }
}

// ============================================================================
// Buffer creation
// ============================================================================
void GPUFluidBackend::createSharedBuffers(const FluidScene& scene) {
    int nx = grid_.gridSize.x, ny = grid_.gridSize.y, nz = grid_.gridSize.z;
    grid_.originX = static_cast<float>(scene.domain.origin.x);
    grid_.originY = static_cast<float>(scene.domain.origin.y);
    grid_.originZ = static_cast<float>(scene.domain.origin.z);

    auto makeStorage = [&](size_t sizeBytes, const char* name) {
        return device_.createBuffer({
            .sizeBytes  = sizeBytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferDst,
            .debugName  = name,
        });
    };

    size_t nU = static_cast<size_t>(nx + 1) * ny * nz;
    size_t nV = static_cast<size_t>(nx) * (ny + 1) * nz;
    size_t nW = static_cast<size_t>(nx) * ny * (nz + 1);

    // Staggered velocity faces (float32)
    grid_.uGrid    = makeStorage(sizeof(float) * nU, "uGrid");
    grid_.vGrid    = makeStorage(sizeof(float) * nV, "vGrid");
    grid_.wGrid    = makeStorage(sizeof(float) * nW, "wGrid");
    grid_.uGridBuf = makeStorage(sizeof(float) * nU, "uGridBuf");
    grid_.vGridBuf = makeStorage(sizeof(float) * nV, "vGridBuf");
    grid_.wGridBuf = makeStorage(sizeof(float) * nW, "wGridBuf");

    // Valid flags (uint32 per face)
    grid_.uValid    = makeStorage(sizeof(uint32_t) * nU, "uValid");
    grid_.vValid    = makeStorage(sizeof(uint32_t) * nV, "vValid");
    grid_.wValid    = makeStorage(sizeof(uint32_t) * nW, "wValid");
    grid_.uValidBuf = makeStorage(sizeof(uint32_t) * nU, "uValidBuf");
    grid_.vValidBuf = makeStorage(sizeof(uint32_t) * nV, "vValidBuf");
    grid_.wValidBuf = makeStorage(sizeof(uint32_t) * nW, "wValidBuf");

    // Particles (SOA: separate position and velocity buffers, tightly-packed float3)
    grid_.particlePositions = device_.createBuffer({
        .sizeBytes  = particleBufferSize(grid_.numParticles),
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        .debugName  = "particlePositions",
    });
    grid_.particleVelocities = device_.createBuffer({
        .sizeBytes  = particleBufferSize(grid_.numParticles),
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferSrc | BufferDesc::TransferDst,
        .debugName  = "particleVelocities",
    });

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
    size_t bufSize = particleBufferSize(grid_.numParticles);

    // Pack SOA arrays (tightly-packed float3)
    std::vector<float> positions(static_cast<size_t>(grid_.numParticles) * 3, 0.0f);
    std::vector<float> velocities(static_cast<size_t>(grid_.numParticles) * 3, 0.0f);
    for (uint32_t i = 0; i < grid_.numParticles; ++i) {
        uint32_t base = i * 3;
        positions[base + 0] = static_cast<float>(scene.initialFluid.positions[i].x);
        positions[base + 1] = static_cast<float>(scene.initialFluid.positions[i].y);
        positions[base + 2] = static_cast<float>(scene.initialFluid.positions[i].z);
        if (i < scene.initialFluid.velocities.size()) {
            velocities[base + 0] = static_cast<float>(scene.initialFluid.velocities[i].x);
            velocities[base + 1] = static_cast<float>(scene.initialFluid.velocities[i].y);
            velocities[base + 2] = static_cast<float>(scene.initialFluid.velocities[i].z);
        }
    }

    // Upload positions via staging buffer
    auto stagingPos = device_.createBuffer({
        .sizeBytes = bufSize,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc,
        .debugName = "positions-staging",
    });
    auto* dstPos = stagingPos->map();
    std::memcpy(dstPos, positions.data(), bufSize);
    stagingPos->unmap();

    // Upload velocities via staging buffer
    auto stagingVel = device_.createBuffer({
        .sizeBytes = bufSize,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc,
        .debugName = "velocities-staging",
    });
    auto* dstVel = stagingVel->map();
    std::memcpy(dstVel, velocities.data(), bufSize);
    stagingVel->unmap();

    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bufSize}}};
    cmd->copyBuffer(stagingPos, grid_.particlePositions, region);
    cmd->copyBuffer(stagingVel, grid_.particleVelocities, region);
    cmd->end();
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

void GPUFluidBackend::uploadColliderToImage(const Mesh& mesh) {
    // Phase 1: stub — collider SDF upload will be implemented with CPU-side
    // mesh-to-SDF conversion, then upload to 3D image via staging buffer.
    std::cout << "[GPU] Collider SDF upload stub (mesh with "
              << mesh.triangleCount << " triangles)\n";
}

// ============================================================================
// Boundary pipelines
// ============================================================================
void GPUFluidBackend::createBoundaryPipelines() {
    psoDirichlet_   = sim::rhi::compileComputePipeline(device_, compiler_, shaderPath("dirichlet.hlsl"));
    psoExtrapolate_ = sim::rhi::compileComputePipeline(device_, compiler_, shaderPath("extrapolate.hlsl"));
    psoBodyForce_   = sim::rhi::compileComputePipeline(device_, compiler_, shaderPath("body-force.hlsl"));
    psoCollider_    = sim::rhi::compileComputePipeline(device_, compiler_, shaderPath("collider.hlsl"));
}

// ============================================================================
// Substep orchestration
// ============================================================================
Real GPUFluidBackend::computeCFL() const {
    // GPU CFL: read back from CPU-side estimate
    // Phase 1: return fixed safe value
    return config_.maxCfl * 0.005;
}

void GPUFluidBackend::substep(CommandList& cmd, Real dt) {
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
    advector_->scatterP2G(cmd, grid_, dt);
    // barrier already done inside scatterP2G

    // ---- 2. Body force (gravity on V faces) ----
    if (psoBodyForce_.valid()) {
        BodyForceParams p;
        p.vGrid      = grid_.vGrid;
        p.dt         = fdt;
        p.gravityY   = static_cast<float>(config_.gravity.y);
        p.numVFaces  = nV;
        cmd.dispatch(psoBodyForce_, p, (nV + 255) / 256, 1, 1);
        barrier();
    }

    // ---- 3. Extrapolate velocities (10 Jacobi steps, ping-pong per component) ----
    if (psoExtrapolate_.valid()) {
        for (int iter = 0; iter < 10; ++iter) {
            // U component
            {
                ExtrapolateParams p;
                p.gridIn   = grid_.uGrid;    p.gridOut  = grid_.uGridBuf;
                p.validIn  = grid_.uValid;   p.validOut = grid_.uValidBuf;
                p.faceResX = nx + 1; p.faceResY = ny; p.faceResZ = nz;
                p.numFaces = nU;
                cmd.dispatch(psoExtrapolate_, p, (nU + 255) / 256, 1, 1);
            }
            // V component
            {
                ExtrapolateParams p;
                p.gridIn   = grid_.vGrid;    p.gridOut  = grid_.vGridBuf;
                p.validIn  = grid_.vValid;   p.validOut = grid_.vValidBuf;
                p.faceResX = nx; p.faceResY = ny + 1; p.faceResZ = nz;
                p.numFaces = nV;
                cmd.dispatch(psoExtrapolate_, p, (nV + 255) / 256, 1, 1);
            }
            // W component
            {
                ExtrapolateParams p;
                p.gridIn   = grid_.wGrid;    p.gridOut  = grid_.wGridBuf;
                p.validIn  = grid_.wValid;   p.validOut = grid_.wValidBuf;
                p.faceResX = nx; p.faceResY = ny; p.faceResZ = nz + 1;
                p.numFaces = nW;
                cmd.dispatch(psoExtrapolate_, p, (nW + 255) / 256, 1, 1);
            }
            barrier();
            // Swap: buf → grid
            std::swap(grid_.uGrid, grid_.uGridBuf);
            std::swap(grid_.vGrid, grid_.vGridBuf);
            std::swap(grid_.wGrid, grid_.wGridBuf);
            std::swap(grid_.uValid, grid_.uValidBuf);
            std::swap(grid_.vValid, grid_.vValidBuf);
            std::swap(grid_.wValid, grid_.wValidBuf);
        }
    }

    // ---- 4. Pressure solve (build weights → build system → CG → project) ----
    projector_->solve(cmd, grid_, dt);
    barrier();

    // ---- 5. G2P gather + advect particles ----
    advector_->gatherAndAdvect(cmd, grid_, dt);
    // final barrier inside gatherAndAdvect
}

} // namespace fluid::gpu
