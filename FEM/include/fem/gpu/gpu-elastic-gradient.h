// ============================================================================
// FEM/include/fem/gpu/gpu-elastic-gradient.h
// GPU Stable Neo-Hookean elastic energy gradient (double, atomic-free).
//
// Self-contained: takes explicit rest/current positions, tet connectivity and
// material (mu, lambda) — decoupled from System. Produces the per-vertex
// elastic energy gradient identical to ElasticTetMesh::assembleEnergyGradient.
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ksk::fem::gpu {

class GpuElasticGradient {
public:
    GpuElasticGradient(ksk::rhi::Device& device,
                       ksk::rhi::ShaderCompiler& compiler,
                       const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Compute elastic energy gradient.
    //   restVerts: rest configuration (X), one glm::dvec3 per vertex
    //   curVerts : current configuration (x)
    //   tets     : tetrahedra connectivity (must be positively oriented)
    //   mu, lambda: Stable Neo-Hookean parameters (already adjusted, as used
    //               by deform::StableNeoHookean)
    //   gradOut  : output gradient, one glm::dvec3 per vertex
    void compute(const std::vector<glm::dvec3>& restVerts,
                 const std::vector<glm::dvec3>& curVerts,
                 const std::vector<std::array<int, 4>>& tets,
                 double mu, double lambda,
                 std::vector<glm::dvec3>& gradOut);

private:
    ksk::rhi::Device& device_;
    bool valid_ = false;
    ksk::rhi::PipelineRef pso_;

    void uploadBytes(const ksk::rhi::BufferRef& dst, const void* data, size_t bytes);
    void download(const ksk::rhi::BufferRef& src, void* dst, size_t bytes);
};

} // namespace ksk::fem::gpu
