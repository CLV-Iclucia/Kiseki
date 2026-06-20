// ============================================================================
// src/gpu/gpu-pcg-solver.cc
// Jacobi-preconditioned CG — fully GPU-resident, float32.
// ============================================================================

#include <FluidSim/gpu/gpu-pcg-solver.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

using namespace sim::rhi;
// ============================================================================
// Constructor
// ============================================================================
GpuPCGSolver::GpuPCGSolver(Device& device, ShaderCompiler& compiler,
                            const PressureSystem& system,
                            const SolverConfig& config,
                            const std::filesystem::path& shaderDir)
    : maxIters_(config.pressureMaxIters)
{
    uint32_t nc           = static_cast<uint32_t>(system.gridSize.x) *
                            system.gridSize.y * system.gridSize.z;
    uint32_t reduceGroups = (nc + 255) / 256;

    auto makeStorage = [&](size_t bytes, const char* name) {
        return device.createBuffer({
            .sizeBytes  = bytes,
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferDst | BufferDesc::TransferSrc,
            .debugName  = name,
        });
    };

    cgR_            = makeStorage(sizeof(float) * nc,           "pcg-r");
    cgZ_            = makeStorage(sizeof(float) * nc,           "pcg-z");
    cgS_            = makeStorage(sizeof(float) * nc,           "pcg-s");
    reduceBuf_      = makeStorage(sizeof(float) * reduceGroups, "pcg-reduce");
    sigmaScalar_    = makeStorage(sizeof(float),                "pcg-sigma");
    dotSZScalar_    = makeStorage(sizeof(float),                "pcg-dotSZ");
    alphaScalar_    = makeStorage(sizeof(float),                "pcg-alpha");
    sigmaNewScalar_ = makeStorage(sizeof(float),                "pcg-sigmaNew");

    psoSpMV_         = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-spmv.hlsl");
    psoDotProduct_   = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-dot-product.hlsl");
    psoReduceFinal_  = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-reduce-final.hlsl");
    psoSaxpy_        = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-saxpy.hlsl");
    psoJacobiPrecond_= sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-jacobi.hlsl");
    psoScalarDiv_    = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-scalar-div.hlsl");
    psoUpdateS_      = sim::rhi::compileComputePipeline(device, compiler, shaderDir / "cg-update-s.hlsl");
}

// ============================================================================
// solve
// ============================================================================
void GpuPCGSolver::solve(CommandList& cmd, const PressureSystem& system) {
    if (!psoSpMV_.valid() || !psoJacobiPrecond_.valid()) return;

    uint32_t nc           = static_cast<uint32_t>(system.gridSize.x) *
                            system.gridSize.y * system.gridSize.z;
    uint32_t cellGroups   = (nc + 255) / 256;
    uint32_t reduceGroups = cellGroups;

    int maxIters = std::min(std::max(maxIters_, 1), 500);

    using B = BarrierDesc;
    auto computeBarrier  = [&]() { cmd.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };
    auto transToCompute  = [&]() { cmd.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                                                     B::AccessTransferWrite,
                                                     B::AccessShaderRead | B::AccessShaderWrite); };

    // ---- Lambda helpers ----
    auto doJacobiPrecond = [&]() {
        PCGJacobiPrecondParams p;
        p.z = cgZ_; p.r = cgR_; p.Adiag = system.Adiag;
        p.active = system.active; p.numCells = nc;
        cmd.dispatch(psoJacobiPrecond_, p, cellGroups, 1, 1);
    };

    auto doDot = [&](BufferRef a, BufferRef bv, BufferRef dst) {
        { PCGDotParams p; p.vecA = a; p.vecB = bv;
          p.reduceBuf = reduceBuf_; p.numCells = nc;
          cmd.dispatch(psoDotProduct_, p, reduceGroups, 1, 1); }
        computeBarrier();
        { PCGReduceFinalParams p; p.reduceBuf = reduceBuf_;
          p.scalarOut = dst; p.numGroups = reduceGroups;
          cmd.dispatch(psoReduceFinal_, p, 1, 1, 1); }
    };

    auto doSpMV = [&](BufferRef src, BufferRef dst) {
        PCGSpMVParams p;
        p.output = dst; p.Adiag = system.Adiag;
        p.Aneighbour0 = system.Aneighbour[0]; p.Aneighbour1 = system.Aneighbour[1];
        p.Aneighbour2 = system.Aneighbour[2]; p.Aneighbour3 = system.Aneighbour[3];
        p.Aneighbour4 = system.Aneighbour[4]; p.Aneighbour5 = system.Aneighbour[5];
        p.src = src;
        p.gridSizeX = system.gridSize.x; p.gridSizeY = system.gridSize.y;
        p.gridSizeZ = system.gridSize.z; p.numCells = nc;
        cmd.dispatch(psoSpMV_, p, cellGroups, 1, 1);
    };

    auto doSaxpy = [&](BufferRef y, BufferRef x, BufferRef alpha, float sign) {
        PCGSaxpyParams p; p.y = y; p.x = x; p.alphaBuf = alpha;
        p.numCells = nc; p.sign = sign;
        cmd.dispatch(psoSaxpy_, p, cellGroups, 1, 1);
    };

    auto doDiv = [&](BufferRef num, BufferRef denom, BufferRef out) {
        PCGScalarDivParams p; p.num = num; p.denom = denom; p.out = out;
        cmd.dispatch(psoScalarDiv_, p, 1, 1, 1);
    };

    auto doUpdateS = [&](BufferRef sigNew, BufferRef sigOld) {
        PCGUpdateSParams p; p.s = cgS_; p.z = cgZ_;
        p.sigmaNewBuf = sigNew; p.sigmaOldBuf = sigOld; p.numCells = nc;
        cmd.dispatch(psoUpdateS_, p, cellGroups, 1, 1);
    };

    // ---- Initialization: p=0, r=rhs, z=M⁻¹r, s=z, σ=dot(z,r) ----
    cmd.fillBuffer(system.pressure, 0u);
    { std::array<BufferCopy,1> r{{{0,0,sizeof(float)*nc}}};
      cmd.copyBuffer(system.rhs, cgR_, r); }
    transToCompute();

    doJacobiPrecond();          // z = r / Adiag
    computeBarrier();
    { std::array<BufferCopy,1> r{{{0,0,sizeof(float)*nc}}};
      cmd.copyBuffer(cgZ_, cgS_, r); }  // s = z
    transToCompute();

    doDot(cgZ_, cgR_, sigmaScalar_);   // σ = dot(z, r)
    computeBarrier();

    // ---- Main PCG loop ----
    for (int iter = 0; iter < maxIters; ++iter) {
        doSpMV(cgS_, cgZ_);                              // z = A*s
        computeBarrier();
        doDot(cgS_, cgZ_, dotSZScalar_);                 // dotSZ = dot(s,z)
        computeBarrier();
        doDiv(sigmaScalar_, dotSZScalar_, alphaScalar_); // α = σ/dotSZ
        computeBarrier();
        doSaxpy(system.pressure, cgS_, alphaScalar_, +1.0f); // p += α*s
        computeBarrier();
        doSaxpy(cgR_, cgZ_, alphaScalar_, -1.0f);            // r -= α*z
        computeBarrier();
        doJacobiPrecond();                               // z = M⁻¹r
        computeBarrier();
        doDot(cgZ_, cgR_, sigmaNewScalar_);              // σ_new = dot(z,r)
        computeBarrier();
        doUpdateS(sigmaNewScalar_, sigmaScalar_);        // s = z + β*s
        computeBarrier();
        { std::array<BufferCopy,1> r{{{0,0,sizeof(float)}}};
          cmd.copyBuffer(sigmaNewScalar_, sigmaScalar_, r); }
        transToCompute();
    }
}

} // namespace fluid::gpu
