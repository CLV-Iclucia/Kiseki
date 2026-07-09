// ============================================================================
// FEM/src/gpu/gpu-gradient-reduce.cc
// Deterministic atomic-free scatter-add of (row, vec3) entries into dense g.
// Pipeline: iota -> radix-sort(row,perm) -> gather vec3 -> flag -> scan ->
//           segstart -> segment-reduce(add into g).
// ============================================================================
#include <fem/gpu/gpu-gradient-reduce.h>

#include <spdlog/spdlog.h>

#include <array>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace sim::fem::gpu {

using namespace sim::rhi;
namespace fs = std::filesystem;

static constexpr uint32_t kWG = 256;
static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

// ============================================================================
// Construction
// ============================================================================
GpuGradientReduce::GpuGradientReduce(Device& device, ShaderCompiler& compiler,
                                     const fs::path& shaderDir)
    : device_(device)
{
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

    sort_ = std::make_unique<sim::rpk::Sort>(device, compiler);

    // iota / flag / segstart are identical to the BCOO sorter's (uint row math).
    psoIota_     = compileComputePipeline(device, compiler, dir / "bcoo-iota.hlsl");
    psoFlag_     = compileComputePipeline(device, compiler, dir / "bcoo-flag.hlsl");
    psoSegStart_ = compileComputePipeline(device, compiler, dir / "bcoo-segstart.hlsl");
    psoGather_   = compileComputePipeline(device, compiler, dir / "grad-gather.hlsl");
    psoReduce_   = compileComputePipeline(device, compiler, dir / "grad-reduce.hlsl");

    valid_ = psoIota_.valid() && psoFlag_.valid() && psoSegStart_.valid() &&
             psoGather_.valid() && psoReduce_.valid();
    if (!valid_)
        spdlog::error("[GpuGradientReduce] pipeline compilation failed");
}

// ============================================================================
// Buffers
// ============================================================================
void GpuGradientReduce::ensureCapacity(uint32_t n) {
    if (n <= cap_) return;
    cap_        = n;
    perm_ = createDeviceLocalBuffer(
        device_, size_t(n) * sizeof(uint32_t), "gr-perm");
    flag_ = createDeviceLocalBuffer(
        device_, size_t(n) * sizeof(uint32_t), "gr-flag");
    segId_ = createDeviceLocalBuffer(
        device_, size_t(n) * sizeof(uint32_t), "gr-segId");
    valScratch_ = createDeviceLocalBuffer(
        device_, size_t(n) * 3 * sizeof(double), "gr-val-scratch");
    segStart_ = createDeviceLocalBuffer(
        device_, size_t(n + 1) * sizeof(uint32_t), "gr-segStart");
}

uint32_t GpuGradientReduce::readbackUint(const BufferRef& src, uint32_t index) {
    auto rb = device_.createBuffer({
        .sizeBytes  = sizeof(uint32_t),
        .visibility = BufferDesc::Visibility::Readback,
        .usage      = BufferDesc::TransferDst,
    });
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{size_t(index) * sizeof(uint32_t), 0, sizeof(uint32_t)}}};
    cmd->copyBuffer(src, rb, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
    uint32_t v = *static_cast<uint32_t*>(rb->map());
    rb->unmap();
    return v;
}

// ============================================================================
// addInto
// ============================================================================
void GpuGradientReduce::addInto(const BufferRef& g, const BufferRef& row,
                                const BufferRef& val, uint32_t n) {
    if (!valid_ || n == 0) return;
    ensureCapacity(n);

    using B = BarrierDesc;
    auto cc = [](CommandList& c) {
        c.memoryBarrier(B::StageComputeShader, B::StageComputeShader);
    };
    const uint32_t gN = groups(n);

    // ---- Phase 1: sort rows, gather values, build compact segments ----
    {
        auto cmd = device_.beginCommands(QueueType::Compute);

        // 1) perm = iota
        { BcooIotaParams p; p.values = perm_; p.n = n;
          cmd->dispatch(psoIota_, p, gN, 1, 1); }
        cc(*cmd);

        // 2) radix sort by row; perm permuted alongside (row sorted in place)
        sort_->pairs(*cmd, row, perm_, n);
        cc(*cmd);

        // 3) gather vec3 values through perm into scratch (out != in)
        { GradGatherParams p; p.valOut = valScratch_; p.valIn = val; p.perm = perm_; p.n = n;
          cmd->dispatch(psoGather_, p, gN, 1, 1); }
        cc(*cmd);

        // 4) segment-start flags over the sorted rows
        { BcooFlagParams p; p.flag = flag_; p.row = row; p.n = n;
          cmd->dispatch(psoFlag_, p, gN, 1, 1); }
        cc(*cmd);

        // 5) inclusive scan of flags -> 1-based segment id per entry
        sort_->scan().inclusive(*cmd, sim::rpk::ScanOp::Sum, sim::rpk::ScalarType::Uint32,
                                flag_, segId_, n);
        cc(*cmd);

        // 6) scatter compact segStart (+ sentinel = n)
        { BcooSegStartParams p; p.segStart = segStart_; p.flag = flag_; p.segId = segId_; p.n = n;
          cmd->dispatch(psoSegStart_, p, gN, 1, 1); }

        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    // numSeg = inclusive-scan value of the last entry
    const uint32_t numSeg = readbackUint(segId_, n - 1);
    if (numSeg == 0) return;

    // ---- Phase 2: segment-reduce the vec3 values, add into g ----
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        GradReduceParams p;
        p.g = g; p.valSorted = valScratch_; p.rowSorted = row;
        p.segStart = segStart_; p.numSeg = numSeg;
        cmd->dispatch(psoReduce_, p, groups(numSeg), 1, 1);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }
}

} // namespace sim::fem::gpu
