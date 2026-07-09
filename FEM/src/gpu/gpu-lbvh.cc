// ============================================================================
// FEM/src/gpu/gpu-lbvh.cc
// GPU Linear BVH construction (Karras 2012).
// ============================================================================
#include <fem/gpu/gpu-lbvh.h>

#include <spdlog/spdlog.h>

namespace sim::fem::gpu
{
    using namespace sim::rhi;

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

    static constexpr uint32_t kWG = 256;
    static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

    IMPLEMENT_COMPUTE_SHADER(
        LBVHMortonCS, FEM_SHADER_DIR "/lbvh-morton.hlsl", "main");
    IMPLEMENT_COMPUTE_SHADER(
        LBVHLeavesCS, FEM_SHADER_DIR "/lbvh-leaves.hlsl", "main");
    IMPLEMENT_COMPUTE_SHADER(
        LBVHInternalCS, FEM_SHADER_DIR "/lbvh-internal.hlsl", "main");
    IMPLEMENT_COMPUTE_SHADER(
        LBVHRefitCS, FEM_SHADER_DIR "/lbvh-refit.hlsl", "main");
    IMPLEMENT_COMPUTE_SHADER(
        LBVHBoundBlockCS, FEM_SHADER_DIR "/lbvh-bound-block.hlsl", "main");
    IMPLEMENT_COMPUTE_SHADER(
        LBVHBoundFinalCS, FEM_SHADER_DIR "/lbvh-bound-final.hlsl", "main");

    GPULBVH::GPULBVH(Device& device, ShaderCompiler& compiler)
        : device_(device),
          morton_(device),
          leaves_(device),
          internal_(device),
          refit_(device),
          boundBlock_(device),
          boundFinal_(device)
    {
        sort_ = std::make_unique<sim::rpk::Sort>(device, compiler);

        valid_ = morton_.valid() && leaves_.valid() &&
            internal_.valid() && refit_.valid() &&
            boundBlock_.valid() && boundFinal_.valid();
        if (!valid_)
            spdlog::error("[GpuLBVH] pipeline compilation failed");
    }

    void GPULBVH::ensureCapacity(uint32_t N)
    {
        if (N <= cap_) return;
        cap_ = N;
        capGroups_ = groups(N);
        const auto nodes = size_t(2 * N - 1);
        keys_ = createDeviceLocalBuffer(device_, size_t(N) * sizeof(uint32_t), "lbvh-keys");
        vals_ = createDeviceLocalBuffer(device_, size_t(N) * sizeof(uint32_t), "lbvh-vals");
        nodeLo_ = createDeviceLocalBuffer(device_, nodes * 3 * sizeof(double), "lbvh-nodeLo");
        nodeHi_ = createDeviceLocalBuffer(device_, nodes * 3 * sizeof(double), "lbvh-nodeHi");
        lch_ = createDeviceLocalBuffer(device_, size_t(N - 1) * sizeof(int32_t), "lbvh-lch");
        rch_ = createDeviceLocalBuffer(device_, size_t(N - 1) * sizeof(int32_t), "lbvh-rch");
        fa_ = createDeviceLocalBuffer(device_, nodes * sizeof(int32_t), "lbvh-fa");
        flags_ = createDeviceLocalBuffer(device_, size_t(N - 1) * sizeof(uint32_t), "lbvh-flags");
        boundPartial_ = createDeviceLocalBuffer(
            device_, size_t(capGroups_) * 6 * sizeof(double), "lbvh-boundPartial");
        if (!sceneBound_)
            sceneBound_ = createDeviceLocalBuffer(
                device_, 6 * sizeof(double), "lbvh-sceneBound");
    }

    void GPULBVH::computeSceneBound(CommandList& cmd, const BufferRef& aabbLo,
                                    const BufferRef& aabbHi, uint32_t N)
    {
        using B = BarrierDesc;
        const uint32_t g = groups(N);
        {
            LBVHBoundBlockCS::Params p;
            p.partial = boundPartial_;
            p.lo = aabbLo;
            p.hi = aabbHi;
            p.n = N;
            cmd.dispatch(boundBlock_, p, g, 1, 1);
        }
        cmd.memoryBarrier(B::StageComputeShader, B::StageComputeShader);
        {
            LBVHBoundFinalCS::Params p;
            p.sceneBound = sceneBound_;
            p.partial = boundPartial_;
            p.numGroups = g;
            cmd.dispatch(boundFinal_, p, 1, 1, 1);
        }
        cmd.memoryBarrier(B::StageComputeShader, B::StageComputeShader);
    }

    void GPULBVH::recordBuild(CommandList& cmd, const BufferRef& aabbLo,
                              const BufferRef& aabbHi, const BufferRef& sceneBound,
                              uint32_t N)
    {
        using B = BarrierDesc;
        auto cc = [](CommandList& c) { c.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };
        auto tc = [](CommandList& c)
        {
            c.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                            B::AccessTransferWrite, B::AccessShaderRead);
        };
        const uint32_t gN = groups(N);
        const uint32_t gI = groups(N - 1);

        cmd.fillBuffer(flags_, 0u);
        tc(cmd);

        {
            LBVHMortonCS::Params p;
            p.keys = keys_;
            p.vals = vals_;
            p.aabbLo = aabbLo;
            p.aabbHi = aabbHi;
            p.sceneBound = sceneBound;
            p.n = N;
            cmd.dispatch(morton_, p, gN, 1, 1);
        }
        cc(cmd);

        sort_->pairs(cmd, keys_, vals_, N);
        cc(cmd);

        {
            LBVHLeavesCS::Params p;
            p.nodeLo = nodeLo_;
            p.nodeHi = nodeHi_;
            p.sortedIdx = vals_;
            p.aabbLo = aabbLo;
            p.aabbHi = aabbHi;
            p.n = N;
            cmd.dispatch(leaves_, p, gN, 1, 1);
        }
        {
            LBVHInternalCS::Params p;
            p.lch = lch_;
            p.rch = rch_;
            p.fa = fa_;
            p.mortons = keys_;
            p.idxs = vals_;
            p.n = N;
            cmd.dispatch(internal_, p, gI, 1, 1);
        }
        cc(cmd);

        {
            LBVHRefitCS::Params p;
            p.nodeLo = nodeLo_;
            p.nodeHi = nodeHi_;
            p.flags = flags_;
            p.lch = lch_;
            p.rch = rch_;
            p.fa = fa_;
            p.n = N;
            cmd.dispatch(refit_, p, gN, 1, 1);
        }
    }

    void GPULBVH::build(const BufferRef& aabbLo, const BufferRef& aabbHi,
                        const BufferRef& sceneBound, uint32_t N)
    {
        if (!valid_ || N < 2) return;
        ensureCapacity(N);
        n_ = N;
        auto cmd = device_.beginCommands(QueueType::Compute);
        recordBuild(*cmd, aabbLo, aabbHi, sceneBound, N);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    void GPULBVH::build(const BufferRef& aabbLo, const BufferRef& aabbHi, uint32_t N)
    {
        if (!valid_ || N < 2) return;
        ensureCapacity(N);
        n_ = N;
        auto cmd = device_.beginCommands(QueueType::Compute);
        computeSceneBound(*cmd, aabbLo, aabbHi, N); // -> sceneBound_
        recordBuild(*cmd, aabbLo, aabbHi, sceneBound_, N);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }
} // namespace sim::fem::gpu
