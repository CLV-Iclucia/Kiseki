// ============================================================================
// src/gpu/gpu-context.cc
// ============================================================================

#include <FluidSim/gpu/gpu-context.h>

namespace fluid::gpu {

GPUFluidContext::GPUFluidContext(sim::rhi::Device& device,
                                 const FluidDomain& dom,
                                 uint32_t nParticles)
    : device_(device)
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

    uint32_t uSize = static_cast<uint32_t>((nx + 1) * ny * nz) * sizeof(float);
    uint32_t vSize = static_cast<uint32_t>(nx * (ny + 1) * nz) * sizeof(float);
    uint32_t wSize = static_cast<uint32_t>(nx * ny * (nz + 1)) * sizeof(float);

    uGrid = createDeviceLocalBuffer(device, uSize, "uGrid");
    uGridBuf = createDeviceLocalBuffer(device, uSize, "uGridBuf");
    vGrid = createDeviceLocalBuffer(device, vSize, "vGrid");
    vGridBuf = createDeviceLocalBuffer(device, vSize, "vGridBuf");
    wGrid = createDeviceLocalBuffer(device, wSize, "wGrid");
    wGridBuf = createDeviceLocalBuffer(device, wSize, "wGridBuf");

    uValid = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx + 1) * ny * nz * sizeof(uint32_t),
        "uValid");
    uValidBuf = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx + 1) * ny * nz * sizeof(uint32_t),
        "uValidBuf");
    vValid = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx) * (ny + 1) * nz * sizeof(uint32_t),
        "vValid");
    vValidBuf = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx) * (ny + 1) * nz * sizeof(uint32_t),
        "vValidBuf");
    wValid = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx) * ny * (nz + 1) * sizeof(uint32_t),
        "wValid");
    wValidBuf = createDeviceLocalBuffer(
        device, static_cast<size_t>(nx) * ny * (nz + 1) * sizeof(uint32_t),
        "wValidBuf");

    particlePositions = createDeviceLocalBuffer(
        device, particleBufferSize(nParticles), "particlePositions");
    particleVelocities = createDeviceLocalBuffer(
        device, particleBufferSize(nParticles), "particleVelocities");

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
