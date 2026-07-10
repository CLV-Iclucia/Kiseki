// ============================================================================
// FEM/src/gpu/gpu-broad-phase.cc
// GPU broad phase via count -> scan -> write.
// ============================================================================
#include <fem/gpu/gpu-broad-phase.h>

#include <spdlog/spdlog.h>

#include <array>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace ksk::fem::gpu {

using namespace ksk::rhi;
namespace fs = std::filesystem;

static constexpr uint32_t kWG = 256;
static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

GpuBroadPhase::GpuBroadPhase(Device& device, ShaderCompiler& compiler, const fs::path& shaderDir)
    : device_(device)
{
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

    scan_ = std::make_unique<ksk::rpk::Scan>(device, compiler);

    psoVtCount_ = compileComputePipeline(device, compiler, dir / "broadphase-vt-count.hlsl");
    psoVtWrite_ = compileComputePipeline(device, compiler, dir / "broadphase-vt-write.hlsl");
    psoEeCount_ = compileComputePipeline(device, compiler, dir / "broadphase-ee-count.hlsl");
    psoEeWrite_ = compileComputePipeline(device, compiler, dir / "broadphase-ee-write.hlsl");

    valid_ = psoVtCount_.valid() && psoVtWrite_.valid() &&
             psoEeCount_.valid() && psoEeWrite_.valid();
    if (!valid_)
        spdlog::error("[GpuBroadPhase] pipeline compilation failed");
}

void GpuBroadPhase::ensureQueryCap(uint32_t numQueries) {
    if (numQueries <= capQuery_) return;
    capQuery_ = numQueries;
    counts_ = createDeviceLocalBuffer(
        device_, size_t(numQueries) * sizeof(uint32_t), "bp-counts");
    offsets_ = createDeviceLocalBuffer(
        device_, size_t(numQueries) * sizeof(uint32_t), "bp-offsets");
}

uint32_t GpuBroadPhase::readbackUintAt(const BufferRef& src, uint32_t index) {
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

uint32_t GpuBroadPhase::runQuery(const PipelineRef& psoCount, const PipelineRef& psoWrite,
                                 const GPULBVH& bvh,
                                 const BufferRef& qLo, const BufferRef& qHi,
                                 const BufferRef& conn, const BufferRef& dHatBuf,
                                 uint32_t numQueries, BufferRef& outBuf, uint32_t& outCap) {
    if (!valid_ || numQueries == 0) return 0;
    const uint32_t N = bvh.numPrims();
    if (N < 2) return 0;
    ensureQueryCap(numQueries);

    using B = BarrierDesc;
    const uint32_t g = groups(numQueries);

    // ---- pass 1: count + exclusive scan ----
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        { BroadCountParams p;
          p.counts = counts_; p.nodeLo = bvh.nodeLo(); p.nodeHi = bvh.nodeHi();
          p.lch = bvh.leftChild(); p.rch = bvh.rightChild(); p.sortedIdx = bvh.sortedIdx();
          p.qLo = qLo; p.qHi = qHi; p.conn = conn; p.dHatBuf = dHatBuf;
          p.numQueries = numQueries; p.numPrims = N;
          cmd->dispatch(psoCount, p, g, 1, 1); }
        cmd->memoryBarrier(B::StageComputeShader, B::StageComputeShader);
        scan_->exclusive(*cmd, ksk::rpk::ScanOp::Sum, ksk::rpk::ScalarType::Uint32,
                         counts_, offsets_, numQueries);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    const uint32_t total = readbackUintAt(offsets_, numQueries - 1) +
                           readbackUintAt(counts_,  numQueries - 1);
    if (total == 0) return 0;

    // ---- grow output, pass 2: write ----
    if (total > outCap) {
        outCap = total;
        outBuf = createDeviceLocalBuffer(
            device_, size_t(total) * 4 * sizeof(uint32_t), "bp-out");
    }
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        { BroadWriteParams p;
          p.out = outBuf; p.nodeLo = bvh.nodeLo(); p.nodeHi = bvh.nodeHi();
          p.lch = bvh.leftChild(); p.rch = bvh.rightChild(); p.sortedIdx = bvh.sortedIdx();
          p.qLo = qLo; p.qHi = qHi; p.conn = conn; p.dHatBuf = dHatBuf; p.offsets = offsets_;
          p.numQueries = numQueries; p.numPrims = N;
          cmd->dispatch(psoWrite, p, g, 1, 1); }
        device_.submitAndWait(*cmd, QueueType::Compute);
    }
    return total;
}

uint32_t GpuBroadPhase::queryVT(const GPULBVH& triBvh, const BufferRef& qVertLo,
                                const BufferRef& qVertHi, const BufferRef& triConn,
                                const BufferRef& dHatBuf, uint32_t numVerts) {
    return runQuery(psoVtCount_, psoVtWrite_, triBvh, qVertLo, qVertHi, triConn, dHatBuf,
                    numVerts, vtOut_, capVt_);
}

uint32_t GpuBroadPhase::queryEE(const GPULBVH& edgeBvh, const BufferRef& qEdgeLo,
                                const BufferRef& qEdgeHi, const BufferRef& edgeConn,
                                const BufferRef& dHatBuf, uint32_t numEdges) {
    return runQuery(psoEeCount_, psoEeWrite_, edgeBvh, qEdgeLo, qEdgeHi, edgeConn, dHatBuf,
                    numEdges, eeOut_, capEe_);
}

} // namespace ksk::fem::gpu
