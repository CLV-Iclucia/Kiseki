// ============================================================================
// FEM/src/gpu/gpu-barrier-assembler.cc
// GPU GIPC barrier gradient + Hessian assembly (one thread per constraint).
// ============================================================================
#include <fem/gpu/gpu-barrier-assembler.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif
#ifndef FEM_INCLUDE_DIR
#define FEM_INCLUDE_DIR "."
#endif

namespace ksk::fem::gpu {

using namespace ksk::rhi;
namespace fs = std::filesystem;

static constexpr uint32_t kWG = 256;
static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

static PipelineRef compileWithShared(Device& device, ShaderCompiler& compiler,
                                     const fs::path& path) {
    ShaderCompileOptions opts;
    opts.entryPoint    = "main";
    opts.includeDirs.push_back(fs::path(FEM_INCLUDE_DIR));
    return compileComputePipeline(device, compiler, path, opts);
}

GpuBarrierAssembler::GpuBarrierAssembler(Device& device, ShaderCompiler& compiler,
                                         const fs::path& shaderDir)
    : device_(device)
{
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;
    pso_ = compileWithShared(device, compiler, dir / "barrier-assemble.hlsl");
    valid_ = pso_.valid();
    if (!valid_)
        spdlog::error("[GpuBarrierAssembler] pipeline compilation failed");
}

void GpuBarrierAssembler::uploadBytes(const BufferRef& dst, const void* data, size_t bytes) {
    if (bytes == 0) return;
    auto staging = device_.createBuffer({
        .sizeBytes  = bytes,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage      = BufferDesc::TransferSrc,
    });
    std::memcpy(staging->map(), data, bytes);
    staging->unmap();
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(staging, dst, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

GpuBarrierAssembler::Result
GpuBarrierAssembler::assemble(const BufferRef& x, const BufferRef& pairs,
                              const std::array<uint32_t, 5>& typeOffsets,
                              uint32_t total, double kappa, double dHat,
                              const BufferRef& xRest) {
    Result res;
    if (!valid_ || total == 0) return res;

    const uint32_t nPP = typeOffsets[1] - typeOffsets[0];
    const uint32_t nPE = typeOffsets[2] - typeOffsets[1];
    const uint32_t nPT = typeOffsets[3] - typeOffsets[2];
    const uint32_t nEE = typeOffsets[4] - typeOffsets[3];

    // Per-kind bucket bases (blocks: 4/9/16/16, grad verts: 2/3/4/4).
    std::array<uint32_t, 4> hessBase{}, gradBase{};
    hessBase[0] = 0;
    hessBase[1] = hessBase[0] + nPP * 4u;
    hessBase[2] = hessBase[1] + nPE * 9u;
    hessBase[3] = hessBase[2] + nPT * 16u;
    const uint32_t Hn = hessBase[3] + nEE * 16u;
    gradBase[0] = 0;
    gradBase[1] = gradBase[0] + nPP * 2u;
    gradBase[2] = gradBase[1] + nPE * 3u;
    gradBase[3] = gradBase[2] + nPT * 4u;
    const uint32_t Gn = gradBase[3] + nEE * 4u;

    res.numHessBlocks  = Hn;
    res.numGradEntries = Gn;
    if (Hn == 0 || Gn == 0) return res;

    // (Re)allocate outputs on growth.
    if (Hn > capH_) {
        capH_ = Hn;
        hessBlocks_ = createDeviceLocalBuffer(
            device_, size_t(Hn) * 9 * sizeof(double), "ba-hblocks");
        hessRow_ = createDeviceLocalBuffer(
            device_, size_t(Hn) * sizeof(uint32_t), "ba-hrow");
        hessCol_ = createDeviceLocalBuffer(
            device_, size_t(Hn) * sizeof(uint32_t), "ba-hcol");
    }
    if (Gn > capG_) {
        capG_ = Gn;
        gradRow_ = createDeviceLocalBuffer(
            device_, size_t(Gn) * sizeof(uint32_t), "ba-grow");
        gradVal_ = createDeviceLocalBuffer(
            device_, size_t(Gn) * 3 * sizeof(double), "ba-gval");
    }
    if (!uParams_)
        uParams_ = createDeviceLocalBuffer(
            device_, 13 * sizeof(uint32_t), "ba-uparams");
    if (!dParams_)
        dParams_ = createDeviceLocalBuffer(
            device_, 2 * sizeof(double), "ba-dparams");

    // uParams = [typeOffsets[0..4], hessBase[0..3], gradBase[0..3]]
    uint32_t up[13];
    for (int i = 0; i < 5; ++i) up[i] = typeOffsets[i];
    for (int i = 0; i < 4; ++i) up[5 + i] = hessBase[i];
    for (int i = 0; i < 4; ++i) up[9 + i] = gradBase[i];
    uploadBytes(uParams_, up, sizeof(up));
    const double dp[2] = {kappa, dHat};
    uploadBytes(dParams_, dp, sizeof(dp));

    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        cmd->memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                           BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderRead);
        BarrierAssembleParams p;
        p.hessBlocks = hessBlocks_; p.hessRow = hessRow_; p.hessCol = hessCol_;
        p.gradRow = gradRow_; p.gradVal = gradVal_;
        p.pairs = pairs; p.x = x; p.uParams = uParams_; p.dParams = dParams_;
        p.xRest = xRest;
        p.total = total;
        cmd->dispatch(pso_, p, groups(total), 1, 1);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    return res;
}

} // namespace ksk::fem::gpu
