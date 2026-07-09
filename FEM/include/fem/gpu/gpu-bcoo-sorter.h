// ============================================================================
// FEM/include/fem/gpu/gpu-bcoo-sorter.h
// GPU-side BCOO row-sort + compact segment-start construction.
//
// Takes an unsorted BCOO matrix (blocks/row/col, duplicates allowed, as produced
// by GPU assembly) and orders it by row, building the compact row-segment offset
// array that the segment-reduce SpMV (GpuBlockPCGSolver) consumes — all on GPU,
// without a CPU round trip. Mirrors maths::BlockSparseMatrix<3>::sortByRow():
//   * sort by row only (column order within a row is irrelevant: duplicates are
//     summed inside a segment by SpMV),
//   * segStart = compact segment starts + trailing sentinel (= nnz),
//   * duplicates are NOT merged (no CSR conversion).
//
// Built on rpk::Sort (radix sort, key=row, value=permutation) + its internal
// rpk::Scan (inclusive prefix sum of segment-start flags).
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/sort.h>

#include <filesystem>
#include <memory>

namespace sim::fem::gpu {

// ---- SHADER_PARAMS ----

SHADER_PARAMS_BEGIN(BcooIotaParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, values);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(BcooGatherParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, blocksOut);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, colOut);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, blocksIn);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, colIn);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, perm);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(BcooFlagParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, flag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, row);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(BcooSegStartParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, segStart);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, flag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, segId);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

// ============================================================================
// GpuBcooSorter
// ============================================================================
class GPUBCOOSorter {
public:
    GPUBCOOSorter(sim::rhi::Device& device,
                  sim::rhi::ShaderCompiler& compiler,
                  const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Sort BCOO by row in place and build the compact segment-start array.
    //   blocks   : double[nnz*9] (column-major dmat3) — permuted in place
    //   row      : uint[nnz]     — sorted ascending in place
    //   col      : uint[nnz]     — permuted in place
    //   segStart : uint[>=nnz+1] — output; first (numSeg+1) entries are valid
    // Submits internally and returns numSeg (number of non-empty row segments).
    uint32_t sort(const sim::rhi::BufferRef& blocks,
                  const sim::rhi::BufferRef& row,
                  const sim::rhi::BufferRef& col,
                  const sim::rhi::BufferRef& segStart,
                  uint32_t nnz);

private:
    sim::rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<sim::rpk::Sort> sort_;   // radix sort + internal scan

    sim::rhi::PipelineRef psoIota_, psoGather_, psoFlag_, psoSegStart_;

    // Scratch (lazily (re)allocated by ensureCapacity)
    sim::rhi::BufferRef perm_, flag_, segId_, blocksScratch_, colScratch_;
    uint32_t cap_ = 0;

    void ensureCapacity(uint32_t nnz);
    uint32_t readbackUint(const sim::rhi::BufferRef& src, uint32_t index);
};

} // namespace sim::fem::gpu
