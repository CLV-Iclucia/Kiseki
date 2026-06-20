// ============================================================================
// include/FluidSim/gpu/gpu-pcg-solver.h
// Jacobi-preconditioned CG pressure solver (fully GPU-resident, float32).
// All scalar operations (alpha, beta) are computed on-GPU without readback.
// ============================================================================
#pragma once

#include <FluidSim/gpu/gpu-pressure-solver.h>

namespace fluid::gpu {

// ---- SHADER_PARAMS (defined in header so .cc can use them) ----

SHADER_PARAMS_BEGIN(PCGSpMVParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, output);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour0);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour1);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour2);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour3);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour4);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Aneighbour5);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, src);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeX);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeY);
    SHADER_PARAM_SCALAR(uint32_t, gridSizeZ);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGDotParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, vecA);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, vecB);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, reduceBuf);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGReduceFinalParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, reduceBuf);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, scalarOut);
    SHADER_PARAM_SCALAR(uint32_t, numGroups);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGSaxpyParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, y);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, x);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, alphaBuf);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
    SHADER_PARAM_SCALAR(float,    sign);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGJacobiPrecondParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, z);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, r);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, Adiag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, active);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGScalarDivParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, num);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, denom);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, out);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(PCGUpdateSParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, s);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, z);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, sigmaNewBuf);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, sigmaOldBuf);
    SHADER_PARAM_SCALAR(uint32_t, numCells);
SHADER_PARAMS_END();

// ============================================================================
class GpuPCGSolver final : public GpuPressureSolver {
public:
    GpuPCGSolver(sim::rhi::Device&         device,
                 sim::rhi::ShaderCompiler& compiler,
                 const PressureSystem&     system,
                 const SolverConfig&       config,
                 const std::filesystem::path& shaderDir);

    void solve(sim::rhi::CommandList& cmd,
               const PressureSystem& system) override;

    void updateConfig(const SolverConfig& config) override {
        maxIters_ = config.pressureMaxIters;
    }

private:
    int maxIters_;

    // CG working vectors
    sim::rhi::BufferRef cgR_, cgZ_, cgS_;

    // Scalar buffers for alpha/beta (GPU-side, no CPU readback)
    sim::rhi::BufferRef reduceBuf_;       // partial sums (one per workgroup)
    sim::rhi::BufferRef sigmaScalar_;     // σ = dot(z, r)
    sim::rhi::BufferRef dotSZScalar_;     // dot(s, z) → denominator for α
    sim::rhi::BufferRef alphaScalar_;     // α = σ / dot(s,z)
    sim::rhi::BufferRef sigmaNewScalar_;  // σ_new for β

    // Pipelines
    sim::rhi::PipelineRef psoSpMV_;
    sim::rhi::PipelineRef psoDotProduct_;
    sim::rhi::PipelineRef psoReduceFinal_;
    sim::rhi::PipelineRef psoSaxpy_;
    sim::rhi::PipelineRef psoJacobiPrecond_;
    sim::rhi::PipelineRef psoScalarDiv_;
    sim::rhi::PipelineRef psoUpdateS_;
};

} // namespace fluid::gpu
