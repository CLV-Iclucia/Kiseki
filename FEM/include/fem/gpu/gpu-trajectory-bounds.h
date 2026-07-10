// ============================================================================
// FEM/include/fem/gpu/gpu-trajectory-bounds.h
// Per-primitive trajectory AABBs for GPU broad phase: each primitive's box is
// the union over its vertices of {x_v, x_v + alpha*p_v}. Output feeds GpuLBVH.
// Works for any primitive arity (triangles: 3, edges: 2) via vertsPerPrim.
// ============================================================================
#pragma once

#include <RHI/rhi.h>

namespace ksk::fem::gpu {

class TrajBoundCS final : public ksk::rhi::ComputeShader<TrajBoundCS> {
public:
    DECLARE_COMPUTE_SHADER(TrajBoundCS);

    SHADER_PARAMS_BEGIN(Params)
        SHADER_PARAM_UAV   (ksk::rhi::BufferRef, outLo);
        SHADER_PARAM_UAV   (ksk::rhi::BufferRef, outHi);
        SHADER_PARAM_SRV   (ksk::rhi::BufferRef, x);
        SHADER_PARAM_SRV   (ksk::rhi::BufferRef, p);
        SHADER_PARAM_SRV   (ksk::rhi::BufferRef, conn);
        SHADER_PARAM_SRV   (ksk::rhi::BufferRef, alphaBuf);
        SHADER_PARAM_SCALAR(uint32_t,            numPrims);
        SHADER_PARAM_SCALAR(uint32_t,            vertsPerPrim);
    SHADER_PARAMS_END();
};

class GpuTrajectoryBounds {
public:
    explicit GpuTrajectoryBounds(ksk::rhi::Device& device);

    [[nodiscard]] bool valid() const { return shader_.valid(); }

    // Records the trajectory-AABB dispatch into `cmd` (no submit), so it can be
    // chained with the LBVH build in a single command list.
    //   x, p      : double[numVerts*3]   (positions, search direction)
    //   conn      : uint[numPrims*vertsPerPrim]
    //   alphaBuf  : double[1]            (step fraction; 0 => static bound)
    //   outLo/outHi: double[numPrims*3]  (output)
    void record(ksk::rhi::CommandList& cmd,
                const ksk::rhi::BufferRef& x,
                const ksk::rhi::BufferRef& p,
                const ksk::rhi::BufferRef& conn,
                const ksk::rhi::BufferRef& alphaBuf,
                const ksk::rhi::BufferRef& outLo,
                const ksk::rhi::BufferRef& outHi,
                uint32_t numPrims, uint32_t vertsPerPrim);

    // Convenience: self-submitting version.
    void compute(const ksk::rhi::BufferRef& x,
                 const ksk::rhi::BufferRef& p,
                 const ksk::rhi::BufferRef& conn,
                 const ksk::rhi::BufferRef& alphaBuf,
                 const ksk::rhi::BufferRef& outLo,
                 const ksk::rhi::BufferRef& outHi,
                 uint32_t numPrims, uint32_t vertsPerPrim);

private:
    ksk::rhi::Device& device_;
    TrajBoundCS shader_;
};

} // namespace ksk::fem::gpu
