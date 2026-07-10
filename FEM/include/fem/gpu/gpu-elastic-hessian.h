// ============================================================================
// FEM/include/fem/gpu/gpu-elastic-hessian.h
// GPU Stable Neo-Hookean elastic Hessian (double): per-tet assembly + Jacobi
// SPD filter + PFPx^T H PFPx, emitting 16 BCOO blocks per tet.
//
// Self-contained (explicit rest/current/tets/material), decoupled from System.
// Output matches ElasticTetMesh::assembleEnergyHessian (16 entries/tet, with
// duplicate (row,col) summed during downstream BCOO sort/merge).
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ksk::fem::gpu {

// Elastic Hessian left resident in device buffers (16*nTets BCOO entries,
// duplicates kept). Suitable for feeding GpuBcooSorter -> GpuBlockPCGSolver
// with no CPU round-trip.
struct DeviceBcoo {
    ksk::rhi::BufferRef blocks;   // double[nnz*9], column-major dmat3
    ksk::rhi::BufferRef row;      // uint[nnz]
    ksk::rhi::BufferRef col;      // uint[nnz]
    uint32_t nnz    = 0;
    uint32_t nVerts = 0;
};

class GpuElasticHessian {
public:
    GpuElasticHessian(ksk::rhi::Device& device,
                      ksk::rhi::ShaderCompiler& compiler,
                      const std::filesystem::path& shaderDir = {},
                      int cyclicSweeps = 20);

    [[nodiscard]] bool valid() const { return valid_; }

    // Emits 16*nTets BCOO entries.
    //   blocksOut: (16M*9) doubles, column-major 3x3 (glm::dmat3 layout)
    //   rowOut/colOut: (16M) global block row/col indices
    void compute(const std::vector<glm::dvec3>& restVerts,
                 const std::vector<glm::dvec3>& curVerts,
                 const std::vector<std::array<int, 4>>& tets,
                 double mu, double lambda,
                 std::vector<double>& blocksOut,
                 std::vector<uint32_t>& rowOut,
                 std::vector<uint32_t>& colOut);

    // Same assembly, but leaves the result in device buffers (owned by this
    // object, reused across calls). No download.
    DeviceBcoo computeToDevice(const std::vector<glm::dvec3>& restVerts,
                               const std::vector<glm::dvec3>& curVerts,
                               const std::vector<std::array<int, 4>>& tets,
                               double mu, double lambda);

private:
    ksk::rhi::Device& device_;
    bool valid_ = false;
    ksk::rhi::PipelineRef pso_;

    // Persistent device output buffers (grown on demand).
    ksk::rhi::BufferRef bBlocks_, bRow_, bCol_;
    uint32_t capEntries_ = 0;

    void uploadBytes(const ksk::rhi::BufferRef& dst, const void* data, size_t bytes);
    void download(const ksk::rhi::BufferRef& src, void* dst, size_t bytes);
};

} // namespace ksk::fem::gpu
