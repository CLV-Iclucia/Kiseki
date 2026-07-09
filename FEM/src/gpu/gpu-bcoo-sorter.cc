// ============================================================================
// FEM/src/gpu/gpu-bcoo-sorter.cc
// GPU BCOO row-sort + compact segment-start construction.
// ============================================================================
#include <fem/gpu/gpu-bcoo-sorter.h>

#include <spdlog/spdlog.h>

#include <array>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace sim::fem::gpu
{
    using namespace sim::rhi;
    namespace fs = std::filesystem;

    static constexpr uint32_t kWG = 256;
    static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

    // ============================================================================
    // Construction
    // ============================================================================
    GPUBCOOSorter::GPUBCOOSorter(Device& device, ShaderCompiler& compiler,
                                 const fs::path& shaderDir)
        : device_(device)
    {
        fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

        sort_ = std::make_unique<rpk::Sort>(device, compiler);

        psoIota_ = compileComputePipeline(device, compiler, dir / "bcoo-iota.hlsl");
        psoGather_ = compileComputePipeline(device, compiler, dir / "bcoo-gather.hlsl");
        psoFlag_ = compileComputePipeline(device, compiler, dir / "bcoo-flag.hlsl");
        psoSegStart_ = compileComputePipeline(device, compiler, dir / "bcoo-segstart.hlsl");

        valid_ = psoIota_.valid() && psoGather_.valid() &&
            psoFlag_.valid() && psoSegStart_.valid();
        if (!valid_)
            spdlog::error("[GpuBcooSorter] pipeline compilation failed");
    }

    // ============================================================================
    // Buffers
    // ============================================================================
    void GPUBCOOSorter::ensureCapacity(uint32_t nnz)
    {
        if (nnz <= cap_) return;
        cap_ = nnz;
        perm_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "bcoo-perm");
        flag_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "bcoo-flag");
        segId_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "bcoo-segId");
        blocksScratch_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * 9 * sizeof(double), "bcoo-blocks-scratch");
        colScratch_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "bcoo-col-scratch");
    }

    uint32_t GPUBCOOSorter::readbackUint(const BufferRef& src, uint32_t index)
    {
        auto rb = device_.createBuffer({
            .sizeBytes = sizeof(uint32_t),
            .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst,
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
    // sort
    // ============================================================================
    uint32_t GPUBCOOSorter::sort(const BufferRef& blocks, const BufferRef& row,
                                 const BufferRef& col, const BufferRef& segStart,
                                 uint32_t nnz)
    {
        if (!valid_ || nnz == 0) return 0;
        ensureCapacity(nnz);

        using B = BarrierDesc;
        auto cc = [](CommandList& c)
        {
            c.memoryBarrier(B::StageComputeShader, B::StageComputeShader);
        };
        auto ct = [](CommandList& c)
        {
            c.memoryBarrier(B::StageComputeShader, B::StageTransfer,
                            B::AccessShaderWrite, B::AccessTransferRead);
        };
        auto tc = [](CommandList& c)
        {
            c.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                            B::AccessTransferWrite, B::AccessShaderRead);
        };
        const uint32_t g = groups(nnz);

        auto cmd = device_.beginCommands(QueueType::Compute);

        // 1) perm = iota
        {
            BcooIotaParams p;
            p.values = perm_;
            p.n = nnz;
            cmd->dispatch(psoIota_, p, g, 1, 1);
        }
        cc(*cmd);

        // 2) radix sort by row; perm permuted alongside (row sorted in place)
        sort_->pairs(*cmd, row, perm_, nnz);
        cc(*cmd);

        // 3) gather blocks/col through perm into scratch (out != in)
        {
            BcooGatherParams p;
            p.blocksOut = blocksScratch_;
            p.colOut = colScratch_;
            p.blocksIn = blocks;
            p.colIn = col;
            p.perm = perm_;
            p.n = nnz;
            cmd->dispatch(psoGather_, p, g, 1, 1);
        }
        ct(*cmd);

        // 4) copy scratch back into the caller's blocks/col (now row-sorted)
        {
            std::array<BufferCopy, 1> rg{{{0, 0, size_t(nnz) * 9 * sizeof(double)}}};
            cmd->copyBuffer(blocksScratch_, blocks, rg);
        }
        {
            std::array<BufferCopy, 1> rg{{{0, 0, size_t(nnz) * sizeof(uint32_t)}}};
            cmd->copyBuffer(colScratch_, col, rg);
        }
        tc(*cmd);

        // 5) segment-start flags over the sorted rows
        {
            BcooFlagParams p;
            p.flag = flag_;
            p.row = row;
            p.n = nnz;
            cmd->dispatch(psoFlag_, p, g, 1, 1);
        }
        cc(*cmd);

        // 6) inclusive scan of flags -> 1-based segment id per entry
        sort_->scan().inclusive(*cmd, sim::rpk::ScanOp::Sum, sim::rpk::ScalarType::Uint32,
                                flag_, segId_, nnz);
        cc(*cmd);

        // 7) scatter compact segStart (+ sentinel = nnz)
        {
            BcooSegStartParams p;
            p.segStart = segStart;
            p.flag = flag_;
            p.segId = segId_;
            p.n = nnz;
            cmd->dispatch(psoSegStart_, p, g, 1, 1);
        }

        device_.submitAndWait(*cmd, QueueType::Compute);

        // numSeg = inclusive-scan value of the last entry
        return readbackUint(segId_, nnz - 1);
    }
} // namespace sim::fem::gpu
