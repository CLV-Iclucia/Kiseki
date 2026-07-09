// ============================================================================
// FEM/include/fem/gpu/gpu-ccd.h
// GPU additive CCD (module D): per-candidate ACCD time-of-impact + global Min
// reduction -> conservative step-size upper bound. The device equivalent of
// CollisionDetector::detect{VertexTriangle,EdgeEdge}Collision feeding
// computeStepSizeUpperBound.
//
// The per-pair ACCD is the SHARED single source <fem/shared/ipc-ccd.h>
// (-> ipc-distance), so the time-of-impact matches the CPU runACCD to within
// the Newton-sqrt precision. Colliders are not handled here yet (deformable
// VT/EE only — same scope as GpuBroadPhase / GpuActivation).
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/reduce.h>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace sim::fem::gpu {

SHADER_PARAMS_BEGIN(CcdParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, toiOut);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, x);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, pdir);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, cand);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, paramBuf);
    SHADER_PARAM_SCALAR(uint32_t,            num);
    SHADER_PARAM_SCALAR(uint32_t,            streamOffset);
SHADER_PARAMS_END();

// ============================================================================
// GpuCcd
// ============================================================================
class GPUACCD {
public:
    GPUACCD(rhi::Device& device,
           rhi::ShaderCompiler& compiler,
           const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Conservative step-size upper bound over all VT/EE candidates.
    //   x      : double[nVerts*3] positions
    //   pdir   : double[nVerts*3] Newton step direction
    //   vtCand : uint[numVt*4] {v,t0,t1,t2}   (may be null if numVt==0)
    //   eeCand : uint[numEe*4] {a0,a1,b0,b1}  (may be null if numEe==0)
    //   toiCap : search cap (typically 1.0); s: ACCD gap fraction (e.g. 0.1)
    // Returns min over candidates of the ACCD toi (== toiCap if no candidate
    // limits the step).
    double stepSizeUpperBound(const rhi::BufferRef& x,
                              const rhi::BufferRef& pdir,
                              const rhi::BufferRef& vtCand, uint32_t numVt,
                              const rhi::BufferRef& eeCand, uint32_t numEe,
                              double toiCap, double s);

    // Per-candidate ACCD toi from the last call: double[numVt+numEe], VT first.
    [[nodiscard]] const rhi::BufferRef& tois() const { return toiOut_; }

private:
    rhi::Device& device_;
    bool valid_ = false;

    // TODO: do we really need a Reduce object for every GPUXXX?
    std::unique_ptr<rpk::Reduce> reduce_;
    rhi::PipelineRef psoVt_, psoEe_;

    rhi::BufferRef toiOut_, result_, param_;
    uint32_t capNc_ = 0;

    void ensureCap(uint32_t nc);
    void uploadBytes(const rhi::BufferRef& dst, const void* data, size_t bytes);
    double readbackDouble(const rhi::BufferRef& src);
};

} // namespace sim::fem::gpu
