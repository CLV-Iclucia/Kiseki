// ============================================================================
// FEM/include/fem/gpu/gpu-block-pcg-solver.h
// GPU-resident Block-Jacobi PCG on BCOO (double). Mirrors CPU BlockPCGSolver.
//
// Sparse format: BCOO, identical to maths::BlockSparseMatrix<3> (SOA):
//   blocks[nnz*9] (column-major dmat3) + rowIdx/colIdx[nnz], duplicates summed.
// SpMV is segment-reduce over row-sorted entries (atomic-free, no CSR merge).
// All PCG scalars live on-GPU; only the residual is read back per chunk.
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-vector.h>

#include <filesystem>

namespace sim::fem::gpu {

// ---- SHADER_PARAMS (header so the .cc can reference them) ----

SHADER_PARAMS_BEGIN(SpmvParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, Ap);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, blocks);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, rowIdx);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, colIdx);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, segStart);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, p);
    SHADER_PARAM_SCALAR(uint32_t,            numSeg);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(JacobiSetupParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, invDiag);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, blocks);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, rowIdx);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, colIdx);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, segStart);
    SHADER_PARAM_SCALAR(uint32_t,            numSeg);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(DotParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, a);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, b);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, partial);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ReduceFinalParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, partial);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, result);
    SHADER_PARAM_SCALAR(uint32_t,            numGroups);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(JacobiApplyDotParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, z);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, partial);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, r);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, invDiag);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(AxpyParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, y);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, x);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, alphaBuf);
    SHADER_PARAM_SCALAR(uint32_t,            n);
    SHADER_PARAM_SCALAR(float,               sign);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(XpbyParams)
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, p);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, z);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, betaBuf);
    SHADER_PARAM_SCALAR(uint32_t,            n);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ScalarDivParams)
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, num);
    SHADER_PARAM_SRV   (sim::rhi::BufferRef, denom);
    SHADER_PARAM_UAV   (sim::rhi::BufferRef, result);
SHADER_PARAMS_END();

// ============================================================================
// GpuBlockPCGSolver
// ============================================================================
class GpuBlockPCGSolver {
public:
    struct Result {
        bool   converged = false;
        int    iters     = 0;
        double residual  = 0.0;   // ||r||
    };

    GpuBlockPCGSolver(sim::rhi::Device& device,
                      sim::rhi::ShaderCompiler& compiler,
                      const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // Solve A x = b with x initialized to zero (x0 = 0).
    // `A` is taken by value and sorted by row internally (duplicates kept).
    Result solve(maths::BlockSparseMatrix<3> A,
                 const maths::BlockVector<3>& b,
                 maths::BlockVector<3>& x,
                 int maxIter, double tol);

    // Solve A x = b entirely from device buffers — no CPU sort / upload.
    // The matrix must already be row-sorted BCOO with compact segStart (e.g.
    // produced by GpuBcooSorter). x0 = 0.
    //   blocks   : double[nnz*9] (column-major dmat3), row-sorted
    //   row/col  : uint[nnz]
    //   segStart : uint[numSeg+1]
    //   bVec     : double[nVerts*3] rhs
    //   xVec     : double[nVerts*3] solution output
    Result solveDevice(const sim::rhi::BufferRef& blocks,
                       const sim::rhi::BufferRef& row,
                       const sim::rhi::BufferRef& col,
                       const sim::rhi::BufferRef& segStart,
                       uint32_t nVerts, uint32_t nnz, uint32_t numSeg,
                       const sim::rhi::BufferRef& bVec,
                       const sim::rhi::BufferRef& xVec,
                       int maxIter, double tol);

private:
    sim::rhi::Device& device_;
    bool valid_ = false;

    // Pipelines
    sim::rhi::PipelineRef psoSpmv_, psoJacobiSetup_, psoDot_, psoReduceFinal_,
                          psoApplyDot_, psoAxpy_, psoXpby_, psoScalarDiv_;

    // Device buffers (lazily (re)allocated by ensureCapacity)
    sim::rhi::BufferRef bBlocks_, bRow_, bCol_, bSeg_;
    sim::rhi::BufferRef vX_, vR_, vZ_, vP_, vAp_, invDiag_, partial_;
    sim::rhi::BufferRef sAlpha_, sBeta_, sPAp_, sRR_, sBB_, sRZ_, sRZnew_;

    uint32_t capVerts_ = 0, capNnz_ = 0, capSeg_ = 0, capGroups_ = 0;

    // Core GPU-resident PCG. Precondition: ensureCapacity() done and vR_ holds b.
    // Uses the passed matrix buffers; leaves the solution in vX_.
    Result runResident(const sim::rhi::BufferRef& blocks,
                       const sim::rhi::BufferRef& row,
                       const sim::rhi::BufferRef& col,
                       const sim::rhi::BufferRef& seg,
                       uint32_t nVerts, uint32_t nnz, uint32_t numSeg,
                       int maxIter, double tol);

    void ensureCapacity(uint32_t nVerts, uint32_t nnz, uint32_t numSeg);
    void uploadBytes(const sim::rhi::BufferRef& dst, const void* data, size_t bytes);
    void copyDeviceVec3(const sim::rhi::BufferRef& src, const sim::rhi::BufferRef& dst,
                        uint32_t nVerts);
    double readbackScalar(const sim::rhi::BufferRef& src);
    void downloadVec3(const sim::rhi::BufferRef& src, maths::BlockVector<3>& out);
};

} // namespace sim::fem::gpu
