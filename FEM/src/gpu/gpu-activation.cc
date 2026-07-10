// ============================================================================
// FEM/src/gpu/gpu-activation.cc
// GPU activation: classify -> stable sort by kind -> scatter.
// ============================================================================
#include <fem/gpu/gpu-activation.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif
#ifndef FEM_INCLUDE_DIR
#define FEM_INCLUDE_DIR "."
#endif

namespace ksk::fem::gpu
{
    using namespace ksk::rhi;
    namespace fs = std::filesystem;

    static constexpr uint32_t kWG = 256;
    static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

    // Compile a kernel that #includes the shared <fem/shared/*.h> headers: add the
    // FEM include root to the compiler's search path (the file's own dir is added
    // automatically by compileHlslFile).
    static PipelineRef compileWithShared(Device& device, ShaderCompiler& compiler,
                                         const fs::path& path)
    {
        ShaderCompileOptions opts;
        opts.entryPoint = "main";
        opts.includeDirs.push_back(fs::path(FEM_INCLUDE_DIR));
        return compileComputePipeline(device, compiler, path, opts);
    }

    GpuActivation::GpuActivation(Device& device, ShaderCompiler& compiler,
                                 const fs::path& shaderDir)
        : device_(device)
    {
        fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

        sort_ = std::make_unique<ksk::rpk::Sort>(device, compiler);

        psoVt_ = compileWithShared(device, compiler, dir / "activation-vt-classify.hlsl");
        psoEe_ = compileWithShared(device, compiler, dir / "activation-ee-classify.hlsl");
        psoScatter_ = compileWithShared(device, compiler, dir / "activation-scatter.hlsl");

        valid_ = psoVt_.valid() && psoEe_.valid() && psoScatter_.valid();
        if (!valid_)
            spdlog::error("[GpuActivation] pipeline compilation failed");
    }

    void GpuActivation::ensureStreamCap(uint32_t nc)
    {
        if (!count_)
            count_ = createDeviceLocalBuffer(device_, 4 * sizeof(uint32_t), "act-count");
        if (nc <= capStream_) return;
        capStream_ = nc;
        key_ = createDeviceLocalBuffer(device_, size_t(nc) * sizeof(uint32_t), "act-key");
        val_ = createDeviceLocalBuffer(device_, size_t(nc) * sizeof(uint32_t), "act-val");
        idxTmp_ = createDeviceLocalBuffer(
            device_, size_t(nc) * 4 * sizeof(int32_t), "act-idxTmp");
    }

    // TODO: this can be abstracted to RHI module
    void GpuActivation::readbackUints(const BufferRef& src, uint32_t n, uint32_t* out)
    {
        auto rb = device_.createBuffer({
            .sizeBytes = size_t(n) * sizeof(uint32_t),
            .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst,
        });
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, size_t(n) * sizeof(uint32_t)}}};
        cmd->copyBuffer(src, rb, region);
        device_.submitAndWait(*cmd, QueueType::Transfer);
        std::memcpy(out, rb->map(), size_t(n) * sizeof(uint32_t));
        rb->unmap();
    }

    GpuActivation::Result
    GpuActivation::activate(const BufferRef& x,
                            const BufferRef& vtCand, uint32_t numVt,
                            const BufferRef& eeCand, uint32_t numEe,
                            const BufferRef& dHatSqrBuf)
    {
        Result res;
        const uint32_t nc = numVt + numEe;
        if (!valid_ || nc == 0) return res;

        ensureStreamCap(nc);

        using B = BarrierDesc;

        // ---- pass 1: classify (VT then EE) + stable sort by kind ----
        {
            auto cmd = device_.beginCommands(QueueType::Compute);
            cmd->fillBuffer(count_, 0u);
            cmd->memoryBarrier(B::StageTransfer, B::StageComputeShader,
                               B::AccessTransferWrite, B::AccessShaderRead | B::AccessShaderWrite);

            if (numVt > 0)
            {
                ActClassifyParams p;
                p.keyOut = key_;
                p.valOut = val_;
                p.idxOut = idxTmp_;
                p.countOut = count_;
                p.x = x;
                p.cand = vtCand;
                p.dHatSqrBuf = dHatSqrBuf;
                p.num = numVt;
                p.streamOffset = 0;
                cmd->dispatch(psoVt_, p, groups(numVt), 1, 1);
            }
            if (numEe > 0)
            {
                ActClassifyParams p;
                p.keyOut = key_;
                p.valOut = val_;
                p.idxOut = idxTmp_;
                p.countOut = count_;
                p.x = x;
                p.cand = eeCand;
                p.dHatSqrBuf = dHatSqrBuf;
                p.num = numEe;
                p.streamOffset = numVt;
                cmd->dispatch(psoEe_, p, groups(numEe), 1, 1);
            }
            cmd->memoryBarrier(B::StageComputeShader, B::StageComputeShader);

            sort_->pairs(*cmd, key_, val_, nc);
            device_.submitAndWait(*cmd, QueueType::Compute);
        }

        // ---- bucket counts -> typeOffsets ----
        uint32_t c[4] = {0, 0, 0, 0};
        readbackUints(count_, 4, c);
        const uint32_t total = c[0] + c[1] + c[2] + c[3];
        res.numConstraints = total;
        res.typeOffsets = {0, c[0], c[0] + c[1], c[0] + c[1] + c[2], total};
        if (total == 0) return res;

        // ---- pass 2: scatter the active prefix into compact output ----
        if (total > capOut_)
        {
            capOut_ = total;
            outIdx_ = createDeviceLocalBuffer(
                device_, size_t(total) * 4 * sizeof(int32_t), "act-out");
        }
        {
            auto cmd = device_.beginCommands(QueueType::Compute);
            ActScatterParams p;
            p.outIdx = outIdx_;
            p.valSorted = val_;
            p.idxTmp = idxTmp_;
            p.total = total;
            cmd->dispatch(psoScatter_, p, groups(total), 1, 1);
            device_.submitAndWait(*cmd, QueueType::Compute);
        }

        return res;
    }
} // namespace ksk::fem::gpu
