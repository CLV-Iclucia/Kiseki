// ============================================================================
// FEM/include/fem/gpu/gpu-lbvh.h
// GPU Linear BVH (Karras 2012): Morton codes -> radix sort (RPK) -> radix-tree
// hierarchy -> bottom-up AABB refit. Bit-compatible with spatify::LBVH<double>
// (10-bit/axis Morton in double; 64-bit (morton<<32|idx) lcp reproduced with two
// 32-bit clz). All outputs live in device buffers for downstream GPU collision.
//
// Node layout (Karras): internal nodes [0, nPrs-1), leaves [nPrs-1, 2*nPrs-1).
// Leaf node (nPrs-1+i) holds sorted primitive sortedIdx[i]. Root = node 0.
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/sort.h>

#include <memory>

namespace sim::fem::gpu {

class LBVHMortonCS final : public sim::rhi::ComputeShader<LBVHMortonCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHMortonCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, keys);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, vals);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, aabbLo);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, aabbHi);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, sceneBound);
        SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAMS_END();
};

class LBVHLeavesCS final : public sim::rhi::ComputeShader<LBVHLeavesCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHLeavesCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, nodeLo);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, nodeHi);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, sortedIdx);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, aabbLo);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, aabbHi);
        SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAMS_END();
};

class LBVHInternalCS final : public sim::rhi::ComputeShader<LBVHInternalCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHInternalCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, lch);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, rch);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, fa);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, mortons);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, idxs);
        SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAMS_END();
};

class LBVHRefitCS final : public sim::rhi::ComputeShader<LBVHRefitCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHRefitCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, nodeLo);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, nodeHi);
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, flags);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, lch);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, rch);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, fa);
        SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAMS_END();
};

class LBVHBoundBlockCS final
    : public sim::rhi::ComputeShader<LBVHBoundBlockCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHBoundBlockCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, partial);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, lo);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, hi);
        SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAMS_END();
};

class LBVHBoundFinalCS final
    : public sim::rhi::ComputeShader<LBVHBoundFinalCS> {
public:
    DECLARE_COMPUTE_SHADER(LBVHBoundFinalCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (sim::rhi::BufferRef, sceneBound);
        SHADER_PARAM_SRV   (sim::rhi::BufferRef, partial);
        SHADER_PARAM_SCALAR(uint32_t,            numGroups);
    SHADER_PARAMS_END();
};

// ============================================================================
// GpuLBVH
// ============================================================================
class GPULBVH {
public:
    GPULBVH(rhi::Device& device,
            rhi::ShaderCompiler& compiler);

    [[nodiscard]] bool valid() const { return valid_; }

    // Build the BVH from N primitive AABBs (device buffers, double, [N*3]).
    //   sceneBound : double[6] = {lo.x,lo.y,lo.z, hi.x,hi.y,hi.z} (union of AABBs)
    // Requires N >= 2. All output buffers are owned by this object.
    void build(const rhi::BufferRef& aabbLo,
               const rhi::BufferRef& aabbHi,
               const rhi::BufferRef& sceneBound,
               uint32_t N);

    // Same, but computes the scene bound on the GPU (reduction over the AABBs),
    // so the BVH is fully device-resident with no host-provided bound.
    void build(const rhi::BufferRef& aabbLo,
               const rhi::BufferRef& aabbHi,
               uint32_t N);

    // --- Outputs (device buffers) ---
    [[nodiscard]] uint32_t numPrims() const { return n_; }
    [[nodiscard]] uint32_t numNodes() const { return n_ ? 2 * n_ - 1 : 0; }
    const rhi::BufferRef& nodeLo()    const { return nodeLo_; }    // double[(2N-1)*3]
    const rhi::BufferRef& nodeHi()    const { return nodeHi_; }    // double[(2N-1)*3]
    const rhi::BufferRef& leftChild() const { return lch_; }       // int[N-1]
    const rhi::BufferRef& rightChild()const { return rch_; }       // int[N-1]
    const rhi::BufferRef& parent()    const { return fa_; }        // int[2N-1]
    const rhi::BufferRef& sortedIdx() const { return vals_; }      // uint[N]
    const rhi::BufferRef& mortons()   const { return keys_; }      // uint[N] (sorted)
    const rhi::BufferRef& sceneBound()const { return sceneBound_; } // double[6]

private:
    rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<rpk::Sort> sort_;

    LBVHMortonCS morton_;
    LBVHLeavesCS leaves_;
    LBVHInternalCS internal_;
    LBVHRefitCS refit_;
    LBVHBoundBlockCS boundBlock_;
    LBVHBoundFinalCS boundFinal_;

    rhi::BufferRef keys_, vals_, nodeLo_, nodeHi_, lch_, rch_, fa_, flags_;
    // TODO: sceneBound_ is a 6-float/double buffer, perhaps can be made into a fixed size buffer, needs RHI support on static array
    rhi::BufferRef boundPartial_, sceneBound_;
    uint32_t cap_ = 0, n_ = 0, capGroups_ = 0;

    void ensureCapacity(uint32_t N);
    void computeSceneBound(rhi::CommandList& cmd,
                           const rhi::BufferRef& aabbLo,
                           const rhi::BufferRef& aabbHi, uint32_t N);
    void recordBuild(rhi::CommandList& cmd,
                     const rhi::BufferRef& aabbLo,
                     const rhi::BufferRef& aabbHi,
                     const rhi::BufferRef& sceneBound, uint32_t N);
};

} // namespace sim::fem::gpu
