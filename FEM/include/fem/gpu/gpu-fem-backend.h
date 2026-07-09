// ============================================================================
// FEM/include/fem/gpu/gpu-fem-backend.h
// GPUFEMBackend — fully GPU-resident FEM backend.
//
// Self-contained: owns its device state (positions, velocities, rest data,
// consistent-mass BCOO) built ONCE from the FEMScene, and runs the whole
// implicit-Euler / Newton step on the GPU by dispatching the device-resident
// shaders directly (elastic-gradient/hessian/energy, mass-apply, vec ops) and
// reusing GpuBcooSorter / GpuBlockPCGSolver / rpk::Reduce as device utilities.
//
// It does NOT use the CPU `System` / `IpcIntegrator`: positions stay resident on
// the device across Newton iterations and across steps; the only host<->device
// transfers are the one-time upload in initialize() and the per-frame readback().
//
// Covers the elastic (Stable Neo-Hookean) implicit-Euler step AND deformable
// self-contact (IPC barrier) with a GPU energy line search. The contact layer
// reuses the existing device components — GpuTrajectoryBounds / GpuLBVH /
// GpuBroadPhase (per-step VT/EE candidates) -> GpuActivation (per-iteration
// PP/PE/PT/EE classification) -> GpuBarrierAssembler (barrier Hessian BCOO +
// gradient entries) -> GpuGradientReduce (fold into RHS) -> GpuCcd (step-size
// upper bound) — and a backend-owned barrier-energy kernel for the line search.
// ============================================================================
#pragma once

#include <fem/fem-backend.h>
#include <fem/gpu/gpu-bcoo-sorter.h>
#include <fem/gpu/gpu-block-pcg-solver.h>
#include <fem/gpu/gpu-trajectory-bounds.h>
#include <fem/gpu/gpu-lbvh.h>
#include <fem/gpu/gpu-broad-phase.h>
#include <fem/gpu/gpu-activation.h>
#include <fem/gpu/gpu-barrier-assembler.h>
#include <fem/gpu/gpu-gradient-reduce.h>
#include <fem/gpu/gpu-ccd.h>
#include <RHI/rhi.h>
#include <RPK/reduce.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace sim::fem {

class GPUFEMBackend : public FEMBackend {
public:
    GPUFEMBackend(sim::rhi::Device& device, sim::rhi::ShaderCompiler& compiler)
        : device_(device), compiler_(compiler) {}

    void initialize(const FEMScene& scene) override;
    void step(Real dt) override;
    void readback(FEMFrame& out) override;

private:
    sim::rhi::Device&         device_;
    sim::rhi::ShaderCompiler& compiler_;
    bool valid_ = false;

    // ---- Device utilities (reused) ----
    std::unique_ptr<gpu::GPUBCOOSorter>     sorter_;
    std::unique_ptr<gpu::GpuBlockPCGSolver> pcg_;
    std::unique_ptr<sim::rpk::Reduce>       reduce_;

    // ---- Contact components (reused; deformable self-contact) ----
    std::unique_ptr<gpu::GpuTrajectoryBounds> traj_;
    std::unique_ptr<gpu::GPULBVH>             triBvh_, edgeBvh_;
    std::unique_ptr<gpu::GpuBroadPhase>       broad_;
    std::unique_ptr<gpu::GpuActivation>       activation_;
    std::unique_ptr<gpu::GpuBarrierAssembler> barrier_;
    std::unique_ptr<gpu::GpuGradientReduce>   gradReduce_;
    std::unique_ptr<gpu::GPUACCD>              ccd_;

    // ---- Backend-owned pipelines (dispatch existing shaders on resident buffers) ----
    sim::rhi::PipelineRef psoEGrad_, psoEHess_, psoEEnergy_, psoBEnergy_, psoMass_,
                          psoAxpy_, psoAbs_, psoMul_;

    // ---- Mesh (host copies for rest-data build + per-frame readback energy) ----
    std::vector<glm::dvec3>        rest_;
    std::vector<std::array<int,4>> tets_;
    std::vector<std::array<int,3>> tris_;   // surface triangles (global indices)
    std::vector<std::array<int,2>> edges_;  // surface edges (global indices)
    uint32_t nVerts_ = 0, nTets_ = 0, nTris_ = 0, nEdges_ = 0;

    double     mu_ = 0.0, lambda_ = 0.0, density_ = 1000.0;
    double     dHat_ = 1e-3, kappa_ = 1e10, eps_ = 1e-2;
    int        pcgMaxIter_ = 1000;
    double     pcgTol_ = 1e-6;
    glm::dvec3 gravity_{0.0, -9.81, 0.0};
    double     time_ = 0.0, lengthScale_ = 1.0;
    int        lastNewtonIters_ = 0;

    // ---- Resident device state, grouped by subsystem (lifetime / ownership) ----
    // Each group is self-contained so a subsystem can be changed in isolation:
    // adding/removing a buffer touches only its group, not a flat dXxx_ pile.

    // Vertex positions, resident across steps & Newton iterations.
    struct Positions {
        sim::rhi::BufferRef rest, cur, vel;   // X / x / xdot   [N*3]
    } pos_;

    // Per-tet rest data + material; constant after initialize().
    struct TetData {
        sim::rhi::BufferRef conn, dmInv, vol, material;       // [M*4]/[M*9]/[M]/[2]
        sim::rhi::BufferRef adjStart, adjTet, adjLocal;       // vertex->incident-tet
    } tet_;

    // Consistent-mass BCOO (16 blocks/tet); constant after initialize().
    struct MassMatrix {
        sim::rhi::BufferRef blocks, row, col;
        uint32_t            nnz = 0;
    } mass_;

    // Newton-iteration scratch (combined-H grows via ensureHCap).
    struct NewtonWork {
        sim::rhi::BufferRef ehBlocks, ehRow, ehCol;           // elastic Hessian (16M)
        sim::rhi::BufferRef hBlocks, hRow, hCol, hSeg;        // combined elastic+barrier+mass BCOO
        sim::rhi::BufferRef gElastic, xhat, diff, mDiff, rhs, p, tmp, elemE;
        sim::rhi::BufferRef scalar;                           // reduction output [1]
        uint32_t            hCap = 0;
    } work_;

    // Deformable self-contact device state (topology const; the rest is scratch).
    struct ContactState {
        sim::rhi::BufferRef triConn, edgeConn, vertConn;      // topology (global idx)
        sim::rhi::BufferRef vertLo, vertHi, triLo, triHi,     // trajectory AABBs
                            edgeLo, edgeHi;
        sim::rhi::BufferRef stepDir;                          // x_hat - x_t (broad-phase traj)
        sim::rhi::BufferRef alpha1, dHat, dHatSqr, bParams;   // scalar/param buffers [1]/[2]
        sim::rhi::BufferRef typeOff;                          // uint[5] typeOffsets
        sim::rhi::BufferRef energy, h2Blocks;                 // per-constraint energy / h^2*barrier blocks
        uint32_t            cap = 0;                          // capacity in candidate count
    } contact_;

    // Helpers
    void uploadBytes(const sim::rhi::BufferRef& dst, const void* data, size_t bytes);
    void downloadBytes(const sim::rhi::BufferRef& src, void* dst, size_t bytes);
    double readbackScalar(const sim::rhi::BufferRef& src);
    void buildDeviceState(const FEMScene& scene);
    void ensureHCap(uint32_t needNnz);          // grow combined-H buffers
    void ensureContactCap(uint32_t numCand);    // grow per-candidate contact buffers
};

// Factory overload (option A): create a GPU-capable backend with explicit
// device/compiler injection. `type` must be "gpu" (or "cpu" for the CPU path).
std::unique_ptr<FEMBackend> createFEMBackend(const std::string& type,
                                             sim::rhi::Device& device,
                                             sim::rhi::ShaderCompiler& compiler);

} // namespace sim::fem
