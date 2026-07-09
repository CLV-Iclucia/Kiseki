// ============================================================================
// src/gpu/gpu-pcg-solver.cc
// Jacobi-preconditioned CG — fully GPU-resident, float32.
// ============================================================================

#include <FluidSim/gpu/gpu-pcg-solver.h>
#include <RHI/rhi.h>

namespace fluid::gpu
{
    using namespace sim::rhi;
    // ============================================================================
    // Constructor
    // ============================================================================
    GpuPCGSolver::GpuPCGSolver(Device& device, const PressureSystem& system,
                               const SolverConfig& config)
        : maxIters_(config.pressureMaxIters),
          spmv_(device),
          dotProduct_(device),
          reduceFinal_(device),
          saxpy_(device),
          jacobiPrecond_(device),
          scalarDiv_(device),
          updateS_(device)
    {
        uint32_t nc = static_cast<uint32_t>(system.gridSize.x) *
            system.gridSize.y * system.gridSize.z;
        uint32_t reduceGroups = (nc + 255) / 256;

        cgR_ = createDeviceLocalBuffer(device, sizeof(float) * nc, "pcg-r");
        cgZ_ = createDeviceLocalBuffer(device, sizeof(float) * nc, "pcg-z");
        cgS_ = createDeviceLocalBuffer(device, sizeof(float) * nc, "pcg-s");
        reduceBuf_ = createDeviceLocalBuffer(
            device, sizeof(float) * reduceGroups, "pcg-reduce");
        sigmaScalar_ = createDeviceLocalBuffer(
            device, sizeof(float), "pcg-sigma");
        dotSZScalar_ = createDeviceLocalBuffer(
            device, sizeof(float), "pcg-dotSZ");
        alphaScalar_ = createDeviceLocalBuffer(
            device, sizeof(float), "pcg-alpha");
        sigmaNewScalar_ = createDeviceLocalBuffer(
            device, sizeof(float), "pcg-sigmaNew");
    }

    // ============================================================================
    // solve
    // ============================================================================
    void GpuPCGSolver::solve(CommandList& cmd, const PressureSystem& system)
    {
        if (!spmv_.valid() || !jacobiPrecond_.valid()) return;

        uint32_t nc = static_cast<uint32_t>(system.gridSize.x) *
            system.gridSize.y * system.gridSize.z;
        uint32_t cellGroups = (nc + 255) / 256;
        uint32_t reduceGroups = cellGroups;

        int maxIters = std::min(std::max(maxIters_, 1), 500);

        using B = BarrierDesc;
        auto computeBarrier = [&]() { cmd.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };
        auto transToCompute = [&]()
        {
            cmd.memoryBarrier(B::StageTransfer, B::StageComputeShader,
                              B::AccessTransferWrite,
                              B::AccessShaderRead | B::AccessShaderWrite);
        };

        // ---- Lambda helpers ----
        auto doJacobiPrecond = [&]()
        {
            PCGJacobiPrecondCS::Params p;
            p.z = cgZ_;
            p.r = cgR_;
            p.Adiag = system.Adiag;
            p.active = system.active;
            p.numCells = nc;
            cmd.dispatch(jacobiPrecond_, p, cellGroups, 1, 1);
        };

        auto doDot = [&](BufferRef a, BufferRef bv, BufferRef dst)
        {
            {
                PCGDotCS::Params p;
                p.vecA = a;
                p.vecB = bv;
                p.reduceBuf = reduceBuf_;
                p.numCells = nc;
                cmd.dispatch(dotProduct_, p, reduceGroups, 1, 1);
            }
            computeBarrier();
            {
                PCGReduceFinalCS::Params p;
                p.reduceBuf = reduceBuf_;
                p.scalarOut = dst;
                p.numGroups = reduceGroups;
                cmd.dispatch(reduceFinal_, p, 1, 1, 1);
            }
        };

        auto doSpMV = [&](BufferRef src, BufferRef dst)
        {
            PCGSpMVCS::Params p;
            p.output = dst;
            p.Adiag = system.Adiag;
            p.Aneighbour0 = system.Aneighbour[0];
            p.Aneighbour1 = system.Aneighbour[1];
            p.Aneighbour2 = system.Aneighbour[2];
            p.Aneighbour3 = system.Aneighbour[3];
            p.Aneighbour4 = system.Aneighbour[4];
            p.Aneighbour5 = system.Aneighbour[5];
            p.src = src;
            p.gridSizeX = system.gridSize.x;
            p.gridSizeY = system.gridSize.y;
            p.gridSizeZ = system.gridSize.z;
            p.numCells = nc;
            cmd.dispatch(spmv_, p, cellGroups, 1, 1);
        };

        auto doSaxpy = [&](BufferRef y, BufferRef x, BufferRef alpha, float sign)
        {
            PCGSaxpyCS::Params p;
            p.y = y;
            p.x = x;
            p.alphaBuf = alpha;
            p.numCells = nc;
            p.sign = sign;
            cmd.dispatch(saxpy_, p, cellGroups, 1, 1);
        };

        auto doDiv = [&](BufferRef num, BufferRef denom, BufferRef dst)
        {
            PCGScalarDivCS::Params p;
            p.num = num;
            p.denom = denom;
            p.result = dst;
            cmd.dispatch(scalarDiv_, p, 1, 1, 1);
        };

        auto doUpdateS = [&](BufferRef sigNew, BufferRef sigOld)
        {
            PCGUpdateSCS::Params p;
            p.s = cgS_;
            p.z = cgZ_;
            p.sigmaNewBuf = sigNew;
            p.sigmaOldBuf = sigOld;
            p.numCells = nc;
            cmd.dispatch(updateS_, p, cellGroups, 1, 1);
        };

        // ---- Initialization: p=0, r=rhs, z=M⁻¹r, s=z, σ=dot(z,r) ----
        cmd.fillBuffer(system.pressure, 0u);
        {
            std::array<BufferCopy, 1> r{{{0, 0, sizeof(float) * nc}}};
            cmd.copyBuffer(system.rhs, cgR_, r);
        }
        transToCompute();

        doJacobiPrecond(); // z = r / Adiag
        computeBarrier();
        {
            std::array<BufferCopy, 1> r{{{0, 0, sizeof(float) * nc}}};
            cmd.copyBuffer(cgZ_, cgS_, r);
        } // s = z
        transToCompute();

        doDot(cgZ_, cgR_, sigmaScalar_); // σ = dot(z, r)
        computeBarrier();

        // ---- Main PCG loop ----
        for (int iter = 0; iter < maxIters; ++iter)
        {
            doSpMV(cgS_, cgZ_); // z = A*s
            computeBarrier();
            doDot(cgS_, cgZ_, dotSZScalar_); // dotSZ = dot(s,z)
            computeBarrier();
            doDiv(sigmaScalar_, dotSZScalar_, alphaScalar_); // α = σ/dotSZ
            computeBarrier();
            doSaxpy(system.pressure, cgS_, alphaScalar_, +1.0f); // p += α*s
            computeBarrier();
            doSaxpy(cgR_, cgZ_, alphaScalar_, -1.0f); // r -= α*z
            computeBarrier();
            doJacobiPrecond(); // z = M⁻¹r
            computeBarrier();
            doDot(cgZ_, cgR_, sigmaNewScalar_); // σ_new = dot(z,r)
            computeBarrier();
            doUpdateS(sigmaNewScalar_, sigmaScalar_); // s = z + β*s
            computeBarrier();
            {
                std::array<BufferCopy, 1> r{{{0, 0, sizeof(float)}}};
                cmd.copyBuffer(sigmaNewScalar_, sigmaScalar_, r);
            }
            transToCompute();
        }
    }
} // namespace fluid::gpu
