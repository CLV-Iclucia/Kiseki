// ============================================================================
// FEM/include/fem/gpu/gpu-energy.h
// GPU evaluation of the barrier-augmented incremental potential energy, the
// missing piece for a fully GPU-resident Newton line search:
//   E(x) = 0.5 (x - x_hat)^T M (x - x_hat) + h^2 ( E_elastic(x) + E_barrier(x) )
//
// Each contribution is computed per-element on the GPU and summed with
// rpk::Reduce(Sum). This first cut implements the elastic Stable Neo-Hookean
// term (energy()/elasticToDevice); the barrier + inertial terms slot in next.
//
// The elastic energy matches CPU ElasticTetMesh::deformationEnergy and is the
// energy whose PK1 / Hessian are produced by GpuElasticGradient/GpuElasticHessian.
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/reduce.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace ksk::fem::gpu {

class GPUEnergy {
public:
    GPUEnergy(rhi::Device& device,
              rhi::ShaderCompiler& compiler,
              const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Total Stable Neo-Hookean elastic energy for the given configuration.
    //   restVerts/curVerts : rest (X) and current (x) positions
    //   tets               : positively oriented tetrahedra
    //   mu, lambda         : SNH parameters (as used by GpuElasticGradient)
    // Returns Σ_tet Psi(F_tet) * restVolume_tet (== CPU deformationEnergy()).
    double elastic(const std::vector<glm::dvec3>& restVerts,
                   const std::vector<glm::dvec3>& curVerts,
                   const std::vector<std::array<int, 4>>& tets,
                   double mu, double lambda);

private:
    rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<rpk::Reduce> reduce_;
    rhi::PipelineRef psoElastic_;

    // Persistent device scratch (grown on demand).
    rhi::BufferRef bElemE_, bResult_;
    rhi::BufferRef bX_, bConn_, bDmInv_, bVol_, bMat_;
    uint32_t capTets_ = 0, capVerts_ = 0;

    void uploadBytes(const rhi::BufferRef& dst, const void* data, size_t bytes);
    double readbackDouble(const rhi::BufferRef& src);
};

} // namespace ksk::fem::gpu
