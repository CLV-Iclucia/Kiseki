// ============================================================================
// src/gpu/gpu-context.cc
// ============================================================================

#include <FluidSim/gpu/gpu-context.h>

namespace fluid::gpu {

GPUFluidContext::GPUFluidContext(sim::rhi::Device& device,
                                 sim::rhi::ShaderCompiler& compiler,
                                 const FluidDomain& dom,
                                 uint32_t nParticles)
    : device_(device), compiler_(compiler)
{
    // 设置基类成员
    domain = dom;
    gridSpacing = dom.size.x / static_cast<Real>(dom.resolution.x);

    gridSize = dom.resolution;
    gpuGridSpacing = static_cast<float>(gridSpacing);
    originX = static_cast<float>(dom.origin.x);
    originY = static_cast<float>(dom.origin.y);
    originZ = static_cast<float>(dom.origin.z);
    numParticles = nParticles;

    int nx = dom.resolution.x;
    int ny = dom.resolution.y;
    int nz = dom.resolution.z;

    using namespace sim::rhi;

    // 速度场 SSBO
    uint32_t uSize = static_cast<uint32_t>((nx + 1) * ny * nz) * sizeof(float);
    uint32_t vSize = static_cast<uint32_t>(nx * (ny + 1) * nz) * sizeof(float);
    uint32_t wSize = static_cast<uint32_t>(nx * ny * (nz + 1)) * sizeof(float);

    BufferDesc bufDesc{};
    bufDesc.usage = BufferDesc::Storage;

    bufDesc.sizeBytes = uSize;
    uGrid = device.createBuffer(bufDesc);
    uGridBuf = device.createBuffer(bufDesc);

    bufDesc.sizeBytes = vSize;
    vGrid = device.createBuffer(bufDesc);
    vGridBuf = device.createBuffer(bufDesc);

    bufDesc.sizeBytes = wSize;
    wGrid = device.createBuffer(bufDesc);
    wGridBuf = device.createBuffer(bufDesc);

    // 有效性标记 SSBO (uint32 per element)
    bufDesc.sizeBytes = static_cast<uint32_t>((nx + 1) * ny * nz) * sizeof(uint32_t);
    uValid = device.createBuffer(bufDesc);
    uValidBuf = device.createBuffer(bufDesc);

    bufDesc.sizeBytes = static_cast<uint32_t>(nx * (ny + 1) * nz) * sizeof(uint32_t);
    vValid = device.createBuffer(bufDesc);
    vValidBuf = device.createBuffer(bufDesc);

    bufDesc.sizeBytes = static_cast<uint32_t>(nx * ny * (nz + 1)) * sizeof(uint32_t);
    wValid = device.createBuffer(bufDesc);
    wValidBuf = device.createBuffer(bufDesc);

    // Particles SSBO (SOA: positions + velocities)
    bufDesc.sizeBytes = particleBufferSize(nParticles);
    particlePositions = device.createBuffer(bufDesc);
    particleVelocities = device.createBuffer(bufDesc);

    // Readback buffers (SOA)
    BufferDesc readbackDesc{};
    readbackDesc.sizeBytes = particleBufferSize(nParticles);
    readbackDesc.visibility = BufferDesc::Visibility::Readback;
    readbackDesc.usage = BufferDesc::TransferDst;
    readbackPositions = device.createBuffer(readbackDesc);
    readbackVelocities = device.createBuffer(readbackDesc);

    // SDF images (3D)
    ImageDesc imgDesc{};
    imgDesc.dim = ImageDesc::Dim::D3;
    imgDesc.width = static_cast<uint32_t>(nx);
    imgDesc.height = static_cast<uint32_t>(ny);
    imgDesc.depth = static_cast<uint32_t>(nz);
    imgDesc.format = Format::R32_Float;
    imgDesc.usage = ImageDesc::Storage | ImageDesc::Sampled;

    colliderSdfImg = device.createImage(imgDesc);
    fluidSdfImg = device.createImage(imgDesc);

    // Sampler for SDF (trilinear)
    SamplerDesc sampDesc{};
    sdfSampler = device.createSampler(sampDesc);
}

void GPUFluidContext::beginFrame() {
    activeCmd_ = device_.beginCommands(sim::rhi::QueueType::Compute);
}

void GPUFluidContext::endFrame() {
    if (activeCmd_) {
        device_.submitAndWait(*activeCmd_, sim::rhi::QueueType::Compute);
        activeCmd_.reset();
    }
}

} // namespace fluid::gpu
