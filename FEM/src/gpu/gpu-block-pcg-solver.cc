// ============================================================================
// FEM/src/gpu/gpu-block-pcg-solver.cc
// GPU-resident Block-Jacobi PCG on BCOO (double).
// ============================================================================
#include <fem/gpu/gpu-block-pcg-solver.h>

#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace sim::fem::gpu {

using namespace sim::rhi;
namespace fs = std::filesystem;

static constexpr uint32_t kWG = 256;
static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

// ============================================================================
// Construction
// ============================================================================
GpuBlockPCGSolver::GpuBlockPCGSolver(Device& device, ShaderCompiler& compiler,
                                     const fs::path& shaderDir)
    : device_(device)
{
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;

    psoSpmv_        = compileComputePipeline(device, compiler, dir / "pcg-spmv.hlsl");
    psoJacobiSetup_ = compileComputePipeline(device, compiler, dir / "pcg-jacobi-setup.hlsl");
    psoDot_         = compileComputePipeline(device, compiler, dir / "pcg-dot.hlsl");
    psoReduceFinal_ = compileComputePipeline(device, compiler, dir / "pcg-reduce-final.hlsl");
    psoApplyDot_    = compileComputePipeline(device, compiler, dir / "pcg-jacobi-apply-dot.hlsl");
    psoAxpy_        = compileComputePipeline(device, compiler, dir / "pcg-axpy.hlsl");
    psoXpby_        = compileComputePipeline(device, compiler, dir / "pcg-xpby.hlsl");
    psoScalarDiv_   = compileComputePipeline(device, compiler, dir / "pcg-scalar-div.hlsl");

    valid_ = psoSpmv_.valid() && psoJacobiSetup_.valid() && psoDot_.valid() &&
             psoReduceFinal_.valid() && psoApplyDot_.valid() && psoAxpy_.valid() &&
             psoXpby_.valid() && psoScalarDiv_.valid();
    if (!valid_)
        spdlog::error("[GpuBlockPCGSolver] pipeline compilation failed");
}

// ============================================================================
// Buffer helpers
// ============================================================================
void GpuBlockPCGSolver::ensureCapacity(uint32_t nVerts, uint32_t nnz, uint32_t numSeg) {
    if (nVerts > capVerts_) {
        capVerts_  = nVerts;
        capGroups_ = groups(nVerts);
        const size_t v3 = size_t(nVerts) * 3 * sizeof(double);
        vX_ = createDeviceLocalBuffer(device_, v3, "pcg-x");
        vR_ = createDeviceLocalBuffer(device_, v3, "pcg-r");
        vZ_ = createDeviceLocalBuffer(device_, v3, "pcg-z");
        vP_ = createDeviceLocalBuffer(device_, v3, "pcg-p");
        vAp_ = createDeviceLocalBuffer(device_, v3, "pcg-Ap");
        invDiag_ = createDeviceLocalBuffer(
            device_, size_t(nVerts) * 9 * sizeof(double), "pcg-invDiag");
        partial_ = createDeviceLocalBuffer(
            device_, size_t(capGroups_) * sizeof(double), "pcg-partial");
    }
    if (nnz > capNnz_) {
        capNnz_  = nnz;
        bBlocks_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * 9 * sizeof(double), "pcg-blocks");
        bRow_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "pcg-row");
        bCol_ = createDeviceLocalBuffer(
            device_, size_t(nnz) * sizeof(uint32_t), "pcg-col");
    }
    if (numSeg > 0 && numSeg + 1 > capSeg_) {
        capSeg_ = numSeg + 1;
        bSeg_ = createDeviceLocalBuffer(
            device_, size_t(capSeg_) * sizeof(uint32_t), "pcg-seg");
    }
    if (!sAlpha_) {
        const size_t s = sizeof(double);
        sAlpha_ = createDeviceLocalBuffer(device_, s, "pcg-alpha");
        sBeta_ = createDeviceLocalBuffer(device_, s, "pcg-beta");
        sPAp_ = createDeviceLocalBuffer(device_, s, "pcg-pAp");
        sRR_ = createDeviceLocalBuffer(device_, s, "pcg-rr");
        sBB_ = createDeviceLocalBuffer(device_, s, "pcg-bb");
        sRZ_ = createDeviceLocalBuffer(device_, s, "pcg-rz");
        sRZnew_ = createDeviceLocalBuffer(device_, s, "pcg-rznew");
    }
}

void GpuBlockPCGSolver::uploadBytes(const BufferRef& dst, const void* data, size_t bytes) {
    if (bytes == 0) return;
    auto staging = device_.createBuffer({
        .sizeBytes = bytes,
        .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc,
    });
    std::memcpy(staging->map(), data, bytes);
    staging->unmap();
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(staging, dst, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

void GpuBlockPCGSolver::copyDeviceVec3(const BufferRef& src, const BufferRef& dst,
                                       uint32_t nVerts) {
    const size_t bytes = size_t(nVerts) * 3 * sizeof(double);
    if (bytes == 0) return;
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(src, dst, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

double GpuBlockPCGSolver::readbackScalar(const BufferRef& src) {
    auto rb = device_.createBuffer({
        .sizeBytes = sizeof(double),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
    });
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
    cmd->copyBuffer(src, rb, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
    double v = *static_cast<double*>(rb->map());
    rb->unmap();
    return v;
}

void GpuBlockPCGSolver::downloadVec3(const BufferRef& src, maths::BlockVector<3>& out) {
    const size_t bytes = size_t(out.numBlocks()) * 3 * sizeof(double);
    auto rb = device_.createBuffer({
        .sizeBytes = bytes,
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
    });
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(src, rb, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
    std::memcpy(out.data(), rb->map(), bytes);
    rb->unmap();
}

// ============================================================================
// solve — host BlockSparseMatrix entry (CPU sort + upload, then runResident)
// ============================================================================
GpuBlockPCGSolver::Result
GpuBlockPCGSolver::solve(maths::BlockSparseMatrix<3> A,
                         const maths::BlockVector<3>& b,
                         maths::BlockVector<3>& x,
                         int maxIter, double tol) {
    Result res;
    if (!valid_) return res;

    // ---- Sort BCOO by row (keeps duplicates) and build segment offsets ----
    A.sortByRow();
    const uint32_t nVerts = static_cast<uint32_t>(b.numBlocks());
    const uint32_t nnz    = static_cast<uint32_t>(A.numEntries());
    if (nnz == 0) return res;

    const auto& rows = A.rowIndices();
    const auto& cols = A.colIndices();
    std::vector<uint32_t> rowU(nnz), colU(nnz);
    std::vector<uint32_t> segStart;
    segStart.reserve(nVerts + 1);
    segStart.push_back(0);
    for (uint32_t k = 0; k < nnz; ++k) {
        rowU[k] = static_cast<uint32_t>(rows[k]);
        colU[k] = static_cast<uint32_t>(cols[k]);
        if (k > 0 && rows[k] != rows[k - 1]) segStart.push_back(k);
    }
    segStart.push_back(nnz);
    const uint32_t numSeg = static_cast<uint32_t>(segStart.size() - 1);

    ensureCapacity(nVerts, nnz, numSeg);

    // ---- Upload matrix + rhs (r = b, x0 = 0) ----
    auto blockBytes =
        asStructuredBytes(std::span<const glm::dmat3>(A.blocks()));
    uploadBytes(bBlocks_, blockBytes.data(), blockBytes.size());
    uploadBytes(bRow_, rowU.data(), size_t(nnz) * sizeof(uint32_t));
    uploadBytes(bCol_, colU.data(), size_t(nnz) * sizeof(uint32_t));
    uploadBytes(bSeg_, segStart.data(), segStart.size() * sizeof(uint32_t));
    uploadBytes(vR_, b.data(), size_t(nVerts) * 3 * sizeof(double));

    res = runResident(bBlocks_, bRow_, bCol_, bSeg_, nVerts, nnz, numSeg, maxIter, tol);
    downloadVec3(vX_, x);
    return res;
}

// ============================================================================
// solveDevice — operate purely on pre-sorted device buffers (no host round trip)
// ============================================================================
GpuBlockPCGSolver::Result
GpuBlockPCGSolver::solveDevice(const BufferRef& blocks, const BufferRef& row,
                               const BufferRef& col, const BufferRef& segStart,
                               uint32_t nVerts, uint32_t nnz, uint32_t numSeg,
                               const BufferRef& bVec, const BufferRef& xVec,
                               int maxIter, double tol) {
    Result res;
    if (!valid_ || nnz == 0) return res;

    ensureCapacity(nVerts, /*nnz*/0, /*numSeg*/0);  // work/scalar buffers only
    copyDeviceVec3(bVec, vR_, nVerts);              // r = b

    res = runResident(blocks, row, col, segStart, nVerts, nnz, numSeg, maxIter, tol);
    copyDeviceVec3(vX_, xVec, nVerts);              // export solution
    return res;
}

// ============================================================================
// runResident — GPU-resident PCG core. Precondition: ensureCapacity done and
// vR_ holds b. Uses the passed matrix buffers; leaves the solution in vX_.
// ============================================================================
GpuBlockPCGSolver::Result
GpuBlockPCGSolver::runResident(const BufferRef& blocks, const BufferRef& row,
                               const BufferRef& col, const BufferRef& seg,
                               uint32_t nVerts, uint32_t nnz, uint32_t numSeg,
                               int maxIter, double tol) {
    Result res;
    (void)nnz;

    const uint32_t vGroups = groups(nVerts);
    const uint32_t sGroups = groups(numSeg);

    using B = BarrierDesc;
    auto cc = [](CommandList& c) { c.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };
    auto tc = [](CommandList& c) { c.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                                                    B::AccessTransferWrite,
                                                    B::AccessShaderRead | B::AccessShaderWrite); };
    auto ct = [](CommandList& c) { c.memoryBarrier(B::StageComputeShader, B::StageTransfer,
                                                    B::AccessShaderWrite, B::AccessTransferRead); };

    auto dot = [&](CommandList& c, BufferRef a, BufferRef bb, BufferRef out) {
        { DotParams p; p.a = a; p.b = bb; p.partial = partial_; p.n = nVerts;
          c.dispatch(psoDot_, p, vGroups, 1, 1); }
        cc(c);
        { ReduceFinalParams p; p.partial = partial_; p.result = out; p.numGroups = vGroups;
          c.dispatch(psoReduceFinal_, p, 1, 1, 1); }
    };
    auto applyDot = [&](CommandList& c, BufferRef out) {
        { JacobiApplyDotParams p; p.z = vZ_; p.partial = partial_; p.r = vR_; p.invDiag = invDiag_;
          p.n = nVerts; c.dispatch(psoApplyDot_, p, vGroups, 1, 1); }
        cc(c);
        { ReduceFinalParams p; p.partial = partial_; p.result = out; p.numGroups = vGroups;
          c.dispatch(psoReduceFinal_, p, 1, 1, 1); }
    };
    auto scalarDiv = [&](CommandList& c, BufferRef num, BufferRef den, BufferRef out) {
        ScalarDivParams p; p.num = num; p.denom = den; p.result = out;
        c.dispatch(psoScalarDiv_, p, 1, 1, 1);
    };

    // ---- Init: invDiag, z = M^-1 r, rz = r.z, p = z, bb = b.b ----
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        tc(*cmd);
        cmd->fillBuffer(vX_, 0u);
        cmd->fillBuffer(vAp_, 0u);
        cmd->fillBuffer(invDiag_, 0u);
        ct(*cmd);

        { JacobiSetupParams p; p.invDiag = invDiag_; p.blocks = blocks; p.rowIdx = row;
          p.colIdx = col; p.segStart = seg; p.numSeg = numSeg;
          cmd->dispatch(psoJacobiSetup_, p, sGroups, 1, 1); }
        cc(*cmd);

        applyDot(*cmd, sRZ_);                 // z = M^-1 r ; rz = r.z
        cc(*cmd);
        dot(*cmd, vR_, vR_, sBB_);            // bb = b.b   (r == b here)
        cc(*cmd);

        // p = z
        ct(*cmd);
        { std::array<BufferCopy, 1> rg{{{0, 0, size_t(nVerts) * 3 * sizeof(double)}}};
          cmd->copyBuffer(vZ_, vP_, rg); }

        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    const double bb = readbackScalar(sBB_);
    const double threshold = tol * tol * bb;  // ||r||^2 / ||b||^2 < tol^2
    if (bb < 1e-300) { res.converged = true; return res; }  // x == 0

    // ---- Chunked PCG loop ----
    const int chunk = 10;
    int iters = 0;
    double rr = bb;
    while (iters < maxIter) {
        const int K = std::min(chunk, maxIter - iters);
        auto cmd = device_.beginCommands(QueueType::Compute);
        for (int it = 0; it < K; ++it) {
            // Ap = A p
            { SpmvParams p; p.Ap = vAp_; p.blocks = blocks; p.rowIdx = row; p.colIdx = col;
              p.segStart = seg; p.p = vP_; p.numSeg = numSeg;
              cmd->dispatch(psoSpmv_, p, sGroups, 1, 1); }
            cc(*cmd);
            dot(*cmd, vP_, vAp_, sPAp_);                       // pAp = p.Ap
            cc(*cmd);
            scalarDiv(*cmd, sRZ_, sPAp_, sAlpha_);             // alpha = rz / pAp
            cc(*cmd);
            { AxpyParams p; p.y = vX_; p.x = vP_; p.alphaBuf = sAlpha_; p.n = nVerts; p.sign = 1.0f;
              cmd->dispatch(psoAxpy_, p, vGroups, 1, 1); }     // x += alpha p
            { AxpyParams p; p.y = vR_; p.x = vAp_; p.alphaBuf = sAlpha_; p.n = nVerts; p.sign = -1.0f;
              cmd->dispatch(psoAxpy_, p, vGroups, 1, 1); }     // r -= alpha Ap
            cc(*cmd);
            dot(*cmd, vR_, vR_, sRR_);                         // rr = r.r
            cc(*cmd);
            applyDot(*cmd, sRZnew_);                           // z = M^-1 r ; rznew = r.z
            cc(*cmd);
            scalarDiv(*cmd, sRZnew_, sRZ_, sBeta_);            // beta = rznew / rz
            cc(*cmd);
            { XpbyParams p; p.p = vP_; p.z = vZ_; p.betaBuf = sBeta_; p.n = nVerts;
              cmd->dispatch(psoXpby_, p, vGroups, 1, 1); }     // p = z + beta p
            cc(*cmd);
            std::swap(sRZ_, sRZnew_);                          // rz <- rznew (handle swap)
        }
        device_.submitAndWait(*cmd, QueueType::Compute);
        iters += K;

        rr = readbackScalar(sRR_);
        if (rr < threshold) { res.converged = true; break; }
    }

    res.iters = iters;
    res.residual = std::sqrt(std::max(rr, 0.0));
    return res;
}

} // namespace sim::fem::gpu
