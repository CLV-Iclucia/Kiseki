// ============================================================================
// FEM/src/gpu/gpu-ccd.cc
// GPU additive CCD: per-candidate ACCD toi -> global Min reduction.
// ============================================================================
#include <fem/gpu/gpu-ccd.h>

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

// Compile a kernel that #includes the shared <fem/shared/*.h> headers.
static PipelineRef compileWithShared(Device& device, ShaderCompiler& compiler,
                                     const fs::path& path) {
    ShaderCompileOptions opts;
    opts.entryPoint    = "main";
    opts.includeDirs.push_back(fs::path(FEM_INCLUDE_DIR));
    return compileComputePipeline(device, compiler, path, opts);
}

GPUACCD::GPUACCD(Device& device, ShaderCompiler& compiler, const fs::path& shaderDir)
    : device_(device)
{
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

    reduce_ = std::make_unique<ksk::rpk::Reduce>(device, compiler);

    psoVt_ = compileWithShared(device, compiler, dir / "ccd-vt.hlsl");
    psoEe_ = compileWithShared(device, compiler, dir / "ccd-ee.hlsl");

    valid_ = psoVt_.valid() && psoEe_.valid();
    if (!valid_)
        spdlog::error("[GpuCcd] pipeline compilation failed");
}

void GPUACCD::uploadBytes(const BufferRef& dst, const void* data, size_t bytes) {
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

double GPUACCD::readbackDouble(const BufferRef& src) {
    auto rb = device_.createBuffer({
        .sizeBytes  = sizeof(double),
        .visibility = BufferDesc::Visibility::Readback,
        .usage      = BufferDesc::TransferDst,
    });
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
    cmd->copyBuffer(src, rb, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
    double v = *static_cast<double*>(rb->map());
    rb->unmap();
    return v;
}

void GPUACCD::ensureCap(uint32_t nc) {
    if (!result_)
        result_ = createDeviceLocalBuffer(device_, sizeof(double), "ccd-result");
    if (!param_)
        param_ = createDeviceLocalBuffer(device_, 2 * sizeof(double), "ccd-param");
    if (nc <= capNc_) return;
    capNc_  = nc;
    toiOut_ = createDeviceLocalBuffer(
        device_, size_t(nc) * sizeof(double), "ccd-toi");
}

double GPUACCD::stepSizeUpperBound(const BufferRef& x, const BufferRef& pdir,
                                  const BufferRef& vtCand, uint32_t numVt,
                                  const BufferRef& eeCand, uint32_t numEe,
                                  double toiCap, double s) {
    const uint32_t nc = numVt + numEe;
    if (!valid_ || nc == 0) return toiCap;

    ensureCap(nc);
    const double params[2] = {toiCap, s};
    uploadBytes(param_, params, sizeof(params));

    using B = BarrierDesc;
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        cmd->memoryBarrier(B::StageTransfer, B::StageComputeShader,
                           B::AccessTransferWrite, B::AccessShaderRead);

        if (numVt > 0) {
            CcdParams p;
            p.toiOut = toiOut_; p.x = x; p.pdir = pdir; p.cand = vtCand; p.paramBuf = param_;
            p.num = numVt; p.streamOffset = 0;
            cmd->dispatch(psoVt_, p, groups(numVt), 1, 1);
        }
        if (numEe > 0) {
            CcdParams p;
            p.toiOut = toiOut_; p.x = x; p.pdir = pdir; p.cand = eeCand; p.paramBuf = param_;
            p.num = numEe; p.streamOffset = numVt;
            cmd->dispatch(psoEe_, p, groups(numEe), 1, 1);
        }
        cmd->memoryBarrier(B::StageComputeShader, B::StageComputeShader);

        reduce_->run(*cmd, ksk::rpk::ReduceOp::Min, ksk::rpk::ScalarType::Float64,
                     toiOut_, result_, nc);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    return readbackDouble(result_);
}

} // namespace ksk::fem::gpu
