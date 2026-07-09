// ============================================================================
// FEM/include/fem/gpu/gpu-gradient-reduce.h
// Deterministic, atomic-free scatter-add of scattered (row, vec3) gradient
// entries into a dense per-vertex gradient buffer g[nVerts*3].
//
// This is the gradient counterpart of GpuBcooSorter: GpuBarrierAssembler emits
// the barrier gradient as scattered per-constraint (vertexRow, dvec3) entries;
// to fold them into the Newton RHS they must be summed per vertex. We do it the
// same way the matrix path does — sort the row keys (rpk::Sort), build compact
// row segments (reusing bcoo-flag/scan/segstart), then segment-reduce the vec3
// values into g. Because each segment maps to a distinct row, the final add is
// race-free without double atomics (which Vulkan lacks) and order-deterministic.
//
// g is read-modify-write: contributions are ADDED on top of the existing
// content, so callers can pre-fill g with the elastic gradient and obtain the
// combined g = g_elastic + g_barrier in place.
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/sort.h>
#include <fem/gpu/gpu-bcoo-sorter.h>   // reuse Bcoo{Iota,Flag,SegStart}Params

#include <filesystem>
#include <memory>

namespace sim::fem::gpu {

// ---- SHADER_PARAMS ----

SHADER_PARAMS_BEGIN(GradGatherParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, valOut);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, valIn);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, perm);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(GradReduceParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, g);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, valSorted);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, rowSorted);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, segStart);
    SHADER_PARAM_SCALAR(uint32_t,            numSeg);
SHADER_PARAMS_END();

// ============================================================================
// GpuGradientReduce
// ============================================================================
class GpuGradientReduce {
public:
    GpuGradientReduce(sim::rhi::Device& device,
                      sim::rhi::ShaderCompiler& compiler,
                      const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Scatter-add n scattered (row, vec3) entries into the dense gradient g.
    //   g   : double[nVerts*3] — read-modify-write (contributions ADDED).
    //   row : uint[n]          — vertex-block index per entry; SORTED IN PLACE.
    //   val : double[n*3]      — the dvec3 contribution per entry (read-only).
    // No-op when n == 0. Submits internally (waits).
    void addInto(const sim::rhi::BufferRef& g,
                 const sim::rhi::BufferRef& row,
                 const sim::rhi::BufferRef& val,
                 uint32_t n);

private:
    sim::rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<sim::rpk::Sort> sort_;   // radix sort + internal scan
    sim::rhi::PipelineRef psoIota_, psoFlag_, psoSegStart_, psoGather_, psoReduce_;

    // Scratch (lazily (re)allocated by ensureCapacity)
    sim::rhi::BufferRef perm_, flag_, segId_, valScratch_, segStart_;
    uint32_t cap_ = 0;

    void ensureCapacity(uint32_t n);
    uint32_t readbackUint(const sim::rhi::BufferRef& src, uint32_t index);
};

} // namespace sim::fem::gpu
