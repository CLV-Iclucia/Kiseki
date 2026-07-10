// ============================================================================
// FEM/include/fem/gpu/gpu-activation.h
// GPU activation (module C): turn broad-phase VT/EE candidates into the unified
// active ConstraintPair set, classified PP/PE/PT/EE with compact typeOffsets —
// the device-side equivalent of IpcIntegrator::refreshActiveConstraintPairs.
//
// Pipeline (count -> stable sort -> scatter):
//   1. classify VT then EE into one combined stream [VT..][EE..]; each active
//      candidate gets key = ConstraintKind (0..3), inactive get 4. Atomic
//      histogram into count[4]. Remapped indices stored per stream slot.
//   2. rpk::Sort::pairs(key, val) — stable -> active entries grouped by kind,
//      VT-before-EE preserved within each kind (matches CPU layout exactly).
//   3. read back count[4] -> typeOffsets; scatter the first `total` sorted
//      entries' indices into the compact output buffer.
//
// All classification/distance/remap math is the SHARED single source in
// <fem/shared/ipc-distance.h> + <fem/shared/ipc-activation.h>, so the result is
// bit-identical to the CPU path. Colliders are not handled here yet (deformable
// VT/EE only — same scope as GpuBroadPhase).
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/sort.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ksk::fem::gpu {

SHADER_PARAMS_BEGIN(ActClassifyParams)
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, keyOut);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, valOut);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, idxOut);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, countOut);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, x);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, cand);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, dHatSqrBuf);
    SHADER_PARAM_SCALAR(uint32_t,            num);
    SHADER_PARAM_SCALAR(uint32_t,            streamOffset);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ActScatterParams)
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, outIdx);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, valSorted);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, idxTmp);
    SHADER_PARAM_SCALAR(uint32_t,            total);
SHADER_PARAMS_END();

// ============================================================================
// GpuActivation
// ============================================================================
class GpuActivation {
public:
    struct Result {
        uint32_t numConstraints = 0;
        // Compact bucket offsets {0, ppEnd, peEnd, ptEnd, total}; matches
        // ConstraintPairSet::typeOffsets (PP, PE, PT, EE, end).
        std::array<uint32_t, 5> typeOffsets = {0, 0, 0, 0, 0};
    };

    GpuActivation(rhi::Device& device,
                  rhi::ShaderCompiler& compiler,
                  const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Classify + bucket the candidates produced by GpuBroadPhase.
    //   x          : double[nVerts*3] positions (BlockVector layout)
    //   vtCand     : uint[numVt*4] {v,t0,t1,t2}   (may be null if numVt==0)
    //   eeCand     : uint[numEe*4] {a0,a1,b0,b1}  (may be null if numEe==0)
    //   dHatSqrBuf : double[1]
    // Output (device) available via pairs(): int[numConstraints*4], the unified
    // ConstraintPair indices (unused slots = -1), grouped per typeOffsets.
    Result activate(const rhi::BufferRef& x,
                    const rhi::BufferRef& vtCand, uint32_t numVt,
                    const rhi::BufferRef& eeCand, uint32_t numEe,
                    const rhi::BufferRef& dHatSqrBuf);

    // int[numConstraints*4] — the activated constraints' vertex-block indices.
    [[nodiscard]] const rhi::BufferRef& pairs() const { return outIdx_; }

private:
    rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<rpk::Sort> sort_;
    rhi::PipelineRef psoVt_, psoEe_, psoScatter_;

    // Combined-stream scratch + output (grown on demand).
    rhi::BufferRef key_, val_, idxTmp_, count_, outIdx_;
    uint32_t capStream_ = 0, capOut_ = 0;

    void ensureStreamCap(uint32_t nc);
    void readbackUints(const rhi::BufferRef& src, uint32_t n, uint32_t* out);
};

} // namespace ksk::fem::gpu
