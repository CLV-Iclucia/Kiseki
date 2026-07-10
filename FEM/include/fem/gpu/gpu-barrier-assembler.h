// ============================================================================
// FEM/include/fem/gpu/gpu-barrier-assembler.h
// GPU GIPC barrier gradient + Hessian assembly (module 2). Consumes the active
// ConstraintPair set from GpuActivation (indices + typeOffsets) and emits, per
// pair, the barrier gradient entries and the rank-1 Hessian BCOO blocks — the
// device equivalent of CPU constraintPairBarrier{Gradient,Hessian}.
//
// All contact math is the SHARED single source (<fem/shared/gipc-pfpx.h> +
// gipc-barrier.h). Output BCOO blocks append to the elastic Hessian for the
// GpuBcooSorter -> GpuBlockPCGSolver path; gradient entries are (row, vec3)
// contributions to be segment-reduced into the global gradient.
//
// Scope: deformable PP/PE/PT/EE (no EE mollifier, no colliders) — matching the
// CPU unified barrier path constraintPairBarrier*.
// ============================================================================
#pragma once

#include <RHI/rhi.h>

#include <array>
#include <cstdint>
#include <filesystem>

namespace ksk::fem::gpu {

SHADER_PARAMS_BEGIN(BarrierAssembleParams)
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, hessBlocks);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, hessRow);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, hessCol);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, gradRow);
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, gradVal);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, pairs);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, x);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, uParams);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, dParams);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, xRest);
    SHADER_PARAM_SCALAR(uint32_t,            total);
SHADER_PARAMS_END();

class GpuBarrierAssembler {
public:
    struct Result {
        uint32_t numHessBlocks = 0;   // PP*4 + PE*9 + (PT+EE)*16
        uint32_t numGradEntries = 0;  // PP*2 + PE*3 + (PT+EE)*4
    };

    GpuBarrierAssembler(ksk::rhi::Device& device,
                        ksk::rhi::ShaderCompiler& compiler,
                        const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Assemble barrier gradient + Hessian from the activated constraints.
    //   x          : double[nVerts*3] positions
    //   pairs      : int[total*4] unified ConstraintPair indices (GpuActivation)
    //   typeOffsets: {0, ppEnd, peEnd, ptEnd, total}
    //   kappa, dHat: barrier stiffness / distance threshold
    //   xRest      : double[nVerts*3] rest positions (EE mollifier eps_x); the
    //                near-parallel edge-edge mollifier is applied when provided.
    Result assemble(const ksk::rhi::BufferRef& x,
                    const ksk::rhi::BufferRef& pairs,
                    const std::array<uint32_t, 5>& typeOffsets,
                    uint32_t total, double kappa, double dHat,
                    const ksk::rhi::BufferRef& xRest);

    // Hessian BCOO (row-major append order); blocks are column-major dmat3.
    [[nodiscard]] const ksk::rhi::BufferRef& hessBlocks() const { return hessBlocks_; }
    [[nodiscard]] const ksk::rhi::BufferRef& hessRow()    const { return hessRow_; }
    [[nodiscard]] const ksk::rhi::BufferRef& hessCol()    const { return hessCol_; }
    // Gradient contributions: row[i] vertex, val[i*3..] the dvec3.
    [[nodiscard]] const ksk::rhi::BufferRef& gradRow() const { return gradRow_; }
    [[nodiscard]] const ksk::rhi::BufferRef& gradVal() const { return gradVal_; }

private:
    ksk::rhi::Device& device_;
    bool valid_ = false;
    ksk::rhi::PipelineRef pso_;

    ksk::rhi::BufferRef hessBlocks_, hessRow_, hessCol_, gradRow_, gradVal_;
    ksk::rhi::BufferRef uParams_, dParams_;
    uint32_t capH_ = 0, capG_ = 0;

    void uploadBytes(const ksk::rhi::BufferRef& dst, const void* data, size_t bytes);
};

} // namespace ksk::fem::gpu
