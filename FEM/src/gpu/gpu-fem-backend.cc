// ============================================================================
// FEM/src/gpu/gpu-fem-backend.cc
// Fully GPU-resident FEM backend (elastic implicit-Euler Newton step).
// No CPU System / IpcIntegrator; all per-iteration compute is on the device.
// ============================================================================
#include <fem/gpu/gpu-fem-backend.h>
#include <fem/cpu/cpu-fem-backend.h>   // for the "cpu" factory branch

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif
#ifndef FEM_INCLUDE_DIR
#define FEM_INCLUDE_DIR "."
#endif

namespace ksk::fem
{
    using namespace ksk::rhi;
    namespace fs = std::filesystem;
    namespace gpu_ns = ksk::fem::gpu;

    static constexpr uint32_t kWG = 256;
    static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

    // ---- SHADER_PARAMS (binding names must match each .hlsl) ----
    SHADER_PARAMS_BEGIN(EGradP)
        SHADER_PARAM_UAV(BufferRef, grad);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SRV(BufferRef, tetConn);
        SHADER_PARAM_SRV(BufferRef, DmInv);
        SHADER_PARAM_SRV(BufferRef, vol);
        SHADER_PARAM_SRV(BufferRef, adjStart);
        SHADER_PARAM_SRV(BufferRef, adjTet);
        SHADER_PARAM_SRV(BufferRef, adjLocal);
        SHADER_PARAM_SRV(BufferRef, material);
        SHADER_PARAM_SCALAR(uint32_t, numVerts);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(EHessP)
        SHADER_PARAM_UAV(BufferRef, outBlocks);
        SHADER_PARAM_UAV(BufferRef, outRow);
        SHADER_PARAM_UAV(BufferRef, outCol);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SRV(BufferRef, tetConn);
        SHADER_PARAM_SRV(BufferRef, DmInv);
        SHADER_PARAM_SRV(BufferRef, vol);
        SHADER_PARAM_SRV(BufferRef, material);
        SHADER_PARAM_SCALAR(uint32_t, numTets);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(EEnergyP)
        SHADER_PARAM_UAV(BufferRef, energyOut);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SRV(BufferRef, tetConn);
        SHADER_PARAM_SRV(BufferRef, DmInv);
        SHADER_PARAM_SRV(BufferRef, vol);
        SHADER_PARAM_SRV(BufferRef, material);
        SHADER_PARAM_SCALAR(uint32_t, numTets);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(MassP)
        SHADER_PARAM_UAV(BufferRef, y);
        SHADER_PARAM_SRV(BufferRef, v);
        SHADER_PARAM_SRV(BufferRef, tetConn);
        SHADER_PARAM_SRV(BufferRef, vol);
        SHADER_PARAM_SRV(BufferRef, adjStart);
        SHADER_PARAM_SRV(BufferRef, adjTet);
        SHADER_PARAM_SRV(BufferRef, adjLocal);
        SHADER_PARAM_SCALAR(double, density);
        SHADER_PARAM_SCALAR(uint32_t, numVerts);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(AxpyP)
        SHADER_PARAM_UAV(BufferRef, y);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SCALAR(double, a);
        SHADER_PARAM_SCALAR(uint32_t, n);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(AbsP)
        SHADER_PARAM_UAV(BufferRef, outv);
        SHADER_PARAM_SRV(BufferRef, inv);
        SHADER_PARAM_SCALAR(uint32_t, n);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(MulP)
        SHADER_PARAM_UAV(BufferRef, outv);
        SHADER_PARAM_SRV(BufferRef, a);
        SHADER_PARAM_SRV(BufferRef, b);
        SHADER_PARAM_SCALAR(uint32_t, n);
    SHADER_PARAMS_END();

    SHADER_PARAMS_BEGIN(BEnergyP)
        SHADER_PARAM_UAV(BufferRef, energyOut);
        SHADER_PARAM_SRV(BufferRef, pairs);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SRV(BufferRef, xRest);
        SHADER_PARAM_SRV(BufferRef, typeOff);
        SHADER_PARAM_SRV(BufferRef, dParams);
        SHADER_PARAM_SCALAR(uint32_t, total);
    SHADER_PARAMS_END();

    // TODO: can be extracted to RHI
    static PipelineRef compileOpts(Device& device, ShaderCompiler& compiler,
                                   const fs::path& path,
                                   const std::vector<std::pair<std::string, std::string>>& defs = {})
    {
        ShaderCompileOptions opts;
        opts.entryPoint = "main";
        opts.defines = defs;
        opts.includeDirs.push_back(fs::path(FEM_INCLUDE_DIR)); // <fem/shared/...>
        return compileComputePipeline(device, compiler, path, opts);
    }

    // ============================================================================
    // Buffer helpers
    // ============================================================================
    // TODO: upload and download can be extracted to RHI
    void GPUFEMBackend::uploadBytes(const BufferRef& dst, const void* data, size_t bytes)
    {
        if (bytes == 0) return;
        auto st = device_.createBuffer({
            .sizeBytes = bytes,
            .visibility = BufferDesc::Visibility::HostVisible, .usage = BufferDesc::TransferSrc
        });
        std::memcpy(st->map(), data, bytes);
        st->unmap();
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(st, dst, rg);
        device_.submitAndWait(*cmd, QueueType::Transfer);
    }

    void GPUFEMBackend::downloadBytes(const BufferRef& src, void* dst, size_t bytes)
    {
        auto rb = device_.createBuffer({
            .sizeBytes = bytes,
            .visibility = BufferDesc::Visibility::Readback, .usage = BufferDesc::TransferDst
        });
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> rg{{{0, 0, bytes}}};
        cmd->copyBuffer(src, rb, rg);
        device_.submitAndWait(*cmd, QueueType::Transfer);
        std::memcpy(dst, rb->map(), bytes);
        rb->unmap();
    }

    // TODO: can be extracted to RHI
    double GPUFEMBackend::readbackScalar(const BufferRef& src)
    {
        double v = 0.0;
        downloadBytes(src, &v, sizeof(double));
        return v;
    }

    // ============================================================================
    // initialize — build all device-resident state from the FEMScene
    // ============================================================================
    void GPUFEMBackend::initialize(const FEMScene& scene)
    {
        fs::path dir = fs::path(FEM_SHADER_DIR);
        // TODO: must unique_ptr?
        sorter_ = std::make_unique<gpu_ns::GPUBCOOSorter>(device_, compiler_);
        pcg_ = std::make_unique<gpu_ns::GpuBlockPCGSolver>(device_, compiler_);
        reduce_ = std::make_unique<ksk::rpk::Reduce>(device_, compiler_);

        // Contact components (deformable self-contact) — reuse the existing ones.
        traj_ = std::make_unique<gpu_ns::GpuTrajectoryBounds>(device_);
        triBvh_ = std::make_unique<gpu_ns::GPULBVH>(device_, compiler_);
        edgeBvh_ = std::make_unique<gpu_ns::GPULBVH>(device_, compiler_);
        broad_ = std::make_unique<gpu_ns::GpuBroadPhase>(device_, compiler_);
        activation_ = std::make_unique<gpu_ns::GpuActivation>(device_, compiler_);
        barrier_ = std::make_unique<gpu_ns::GpuBarrierAssembler>(device_, compiler_);
        gradReduce_ = std::make_unique<gpu_ns::GpuGradientReduce>(device_, compiler_);
        ccd_ = std::make_unique<gpu_ns::GPUACCD>(device_, compiler_);

        psoEGrad_ = compileOpts(device_, compiler_, dir / "elastic-gradient.hlsl");
        psoEHess_ = compileOpts(device_, compiler_, dir / "elastic-hessian.hlsl",
                                {{"JACOBI_SWEEPS", "20"}});
        psoEEnergy_ = compileOpts(device_, compiler_, dir / "energy-elastic.hlsl");
        psoBEnergy_ = compileOpts(device_, compiler_, dir / "energy-barrier.hlsl");
        psoMass_ = compileOpts(device_, compiler_, dir / "mass-apply.hlsl");
        psoAxpy_ = compileOpts(device_, compiler_, dir / "vec-axpy.hlsl");
        psoAbs_ = compileOpts(device_, compiler_, dir / "vec-abs.hlsl");
        psoMul_ = compileOpts(device_, compiler_, dir / "vec-mul.hlsl");

        // TODO: no need, can validate after every compilation
        valid_ = sorter_->valid() && pcg_->valid() &&
            traj_->valid() && triBvh_->valid() && edgeBvh_->valid() &&
            broad_->valid() && activation_->valid() && barrier_->valid() &&
            gradReduce_->valid() && ccd_->valid() &&
            psoEGrad_.valid() && psoEHess_.valid() && psoEEnergy_.valid() &&
            psoBEnergy_.valid() && psoMass_.valid() && psoAxpy_.valid() &&
            psoAbs_.valid() && psoMul_.valid();
        if (!valid_)
        {
            spdlog::error("[GPUFEMBackend] pipeline/util init failed");
            return;
        }

        // --- material / params (first mesh; uniform material assumption) ---
        // TODO: every tet mesh can have its own constitution
        const auto& m0 = scene.meshes.at(0);
        double E = m0.youngModulus, nu = m0.poissonRatio;
        double muRaw = E / (2.0 * (1.0 + nu));
        double lamRaw = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        mu_ = (0.4 / 0.3) * muRaw; // Stable Neo-Hookean adjustment
        lambda_ = lamRaw + (0.5 / 0.6) * muRaw; // (matches deform::StableNeoHookean)
        density_ = m0.density;
        dHat_ = scene.dHat;
        kappa_ = scene.contactStiffness;
        eps_ = scene.convergenceEps;
        pcgMaxIter_ = scene.pcgMaxIter;
        pcgTol_ = scene.pcgTolerance;
        gravity_ = scene.gravity;
        time_ = 0.0;

        // --- concatenate meshes (rest X, initial x, initial v, tets, surface) ---
        rest_.clear();
        tets_.clear();
        tris_.clear();
        edges_.clear();
        std::vector<glm::dvec3> x0, v0;
        for (const auto& md : scene.meshes)
        {
            int base = int(rest_.size());
            for (auto& X : md.vertices) rest_.push_back(X);
            for (size_t i = 0; i < md.vertices.size(); ++i)
            {
                x0.push_back(md.initialPositions.empty()
                                 ? md.vertices[i]
                                 : md.initialPositions[i]);
                v0.push_back(md.initialVelocities.empty()
                                 ? glm::dvec3(0.0)
                                 : md.initialVelocities[i]);
            }
            for (auto& t : md.tets)
                tets_.push_back({t[0] + base, t[1] + base, t[2] + base, t[3] + base});
            for (auto& tri : md.triangles)
                tris_.push_back({tri[0] + base, tri[1] + base, tri[2] + base});
            for (auto& e : md.edges)
                edges_.push_back({e[0] + base, e[1] + base});
        }
        nVerts_ = uint32_t(rest_.size());
        nTets_ = uint32_t(tets_.size());
        nTris_ = uint32_t(tris_.size());
        nEdges_ = uint32_t(edges_.size());

        // length scale = rest bbox diagonal
        glm::dvec3 lo(1e300), hi(-1e300);
        for (auto& X : rest_)
        {
            lo = glm::min(lo, X);
            hi = glm::max(hi, X);
        }
        lengthScale_ = glm::length(hi - lo);
        if (!(lengthScale_ > 0.0)) lengthScale_ = 1.0;

        buildDeviceState(scene);

        // Upload the initial deformed config (x) and velocity (v) into the resident
        // buffers allocated by buildDeviceState().
        auto xBytes = asStructuredBytes(std::span<const glm::dvec3>(x0));
        auto vBytes = asStructuredBytes(std::span<const glm::dvec3>(v0));
        uploadBytes(pos_.cur, xBytes.data(), xBytes.size());
        uploadBytes(pos_.vel, vBytes.data(), vBytes.size());
    }

    void GPUFEMBackend::buildDeviceState(const FEMScene&)
    {
        const uint32_t N = nVerts_, M = nTets_;

        // rest-data: DmInv (col-major), vol, conn
        std::vector<glm::dmat3> dmInv(M);
        std::vector<double> vol(M);
        std::vector<uint32_t> conn(size_t(M) * 4);
        // TODO: should be computed on GPU
        for (uint32_t t = 0; t < M; ++t)
        {
            const auto& tet = tets_[t];
            glm::dvec3 p0 = rest_[tet[0]];
            glm::dmat3 Dm(rest_[tet[1]] - p0, rest_[tet[2]] - p0, rest_[tet[3]] - p0);
            dmInv[t] = glm::inverse(Dm);
            vol[t] = glm::determinant(Dm) / 6.0;
            for (int j = 0; j < 4; ++j) conn[size_t(t) * 4 + j] = uint32_t(tet[j]);
        }
        // adjacency (vertex -> incident (tet, local))
        std::vector<uint32_t> adjStart(N + 1, 0);
        for (uint32_t t = 0; t < M; ++t) for (int j = 0; j < 4; ++j) adjStart[tets_[t][j] + 1]++;
        for (uint32_t i = 0; i < N; ++i) adjStart[i + 1] += adjStart[i];
        const uint32_t K = adjStart[N];
        std::vector<uint32_t> adjTet(K), adjLocal(K), cursor(adjStart.begin(), adjStart.end());
        for (uint32_t t = 0; t < M; ++t)
            for (int j = 0; j < 4; ++j)
            {
                uint32_t v = uint32_t(tets_[t][j]);
                uint32_t s = cursor[v]++;
                adjTet[s] = t;
                adjLocal[s] = uint32_t(j);
            }
        // consistent mass BCOO (16/tet)
        std::vector<double> mBlk(size_t(M) * 16 * 9, 0.0);
        std::vector<uint32_t> mRow(size_t(M) * 16), mCol(size_t(M) * 16);
        for (uint32_t t = 0; t < M; ++t)
        {
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                {
                    uint32_t e = t * 16 + uint32_t(j * 4 + k);
                    double coeff = density_ * vol[t] * (j == k ? 0.1 : 0.05);
                    mRow[e] = uint32_t(tets_[t][j]);
                    mCol[e] = uint32_t(tets_[t][k]);
                    mBlk[size_t(e) * 9 + 0] = coeff;
                    mBlk[size_t(e) * 9 + 4] = coeff;
                    mBlk[size_t(e) * 9 + 8] = coeff;
                }
        }
        mass_.nnz = M * 16;

        // ---- allocate + upload constant device state ----
        pos_.rest = createDeviceLocalBuffer(
            device_, size_t(N) * 3 * sizeof(double), "fem-X");
        pos_.cur = createDeviceLocalBuffer(
            device_, size_t(N) * 3 * sizeof(double), "fem-x");
        pos_.vel = createDeviceLocalBuffer(
            device_, size_t(N) * 3 * sizeof(double), "fem-v");
        tet_.conn = createDeviceLocalBuffer(
            device_, size_t(M) * 4 * sizeof(uint32_t), "fem-conn");
        tet_.dmInv = createDeviceLocalBuffer(
            device_, size_t(M) * 9 * sizeof(double), "fem-dminv");
        tet_.vol = createDeviceLocalBuffer(
            device_, size_t(M) * sizeof(double), "fem-vol");
        tet_.material = createDeviceLocalBuffer(
            device_, 2 * sizeof(double), "fem-mat");
        tet_.adjStart = createDeviceLocalBuffer(
            device_, size_t(N + 1) * sizeof(uint32_t), "fem-adjs");
        tet_.adjTet = createDeviceLocalBuffer(
            device_, size_t(K) * sizeof(uint32_t), "fem-adjt");
        tet_.adjLocal = createDeviceLocalBuffer(
            device_, size_t(K) * sizeof(uint32_t), "fem-adjl");
        mass_.blocks = createDeviceLocalBuffer(
            device_, size_t(mass_.nnz) * 9 * sizeof(double), "fem-mblk");
        mass_.row = createDeviceLocalBuffer(
            device_, size_t(mass_.nnz) * sizeof(uint32_t), "fem-mrow");
        mass_.col = createDeviceLocalBuffer(
            device_, size_t(mass_.nnz) * sizeof(uint32_t), "fem-mcol");

        auto restBytes = asStructuredBytes(std::span<const glm::dvec3>(rest_));
        auto dmInvBytes = asStructuredBytes(std::span<const glm::dmat3>(dmInv));
        uploadBytes(pos_.rest, restBytes.data(), restBytes.size());
        uploadBytes(tet_.conn, conn.data(), conn.size() * sizeof(uint32_t));
        uploadBytes(tet_.dmInv, dmInvBytes.data(), dmInvBytes.size());
        uploadBytes(tet_.vol, vol.data(), vol.size() * sizeof(double));
        const double mat[2] = {mu_, lambda_};
        uploadBytes(tet_.material, mat, sizeof(mat));
        uploadBytes(tet_.adjStart, adjStart.data(), adjStart.size() * sizeof(uint32_t));
        uploadBytes(tet_.adjTet, adjTet.data(), adjTet.size() * sizeof(uint32_t));
        uploadBytes(tet_.adjLocal, adjLocal.data(), adjLocal.size() * sizeof(uint32_t));
        uploadBytes(mass_.blocks, mBlk.data(), mBlk.size() * sizeof(double));
        uploadBytes(mass_.row, mRow.data(), mRow.size() * sizeof(uint32_t));
        uploadBytes(mass_.col, mCol.data(), mCol.size() * sizeof(uint32_t));

        // ---- Newton work buffers ----
        const uint32_t v3 = N * 3;
        const uint32_t ehNnz = M * 16, combNnz = M * 32;
        work_.hCap = combNnz;
        work_.ehBlocks = createDeviceLocalBuffer(
            device_, size_t(ehNnz) * 9 * sizeof(double), "fem-ehblk");
        work_.ehRow = createDeviceLocalBuffer(
            device_, size_t(ehNnz) * sizeof(uint32_t), "fem-ehrow");
        work_.ehCol = createDeviceLocalBuffer(
            device_, size_t(ehNnz) * sizeof(uint32_t), "fem-ehcol");
        work_.hBlocks = createDeviceLocalBuffer(
            device_, size_t(combNnz) * 9 * sizeof(double), "fem-hblk");
        work_.hRow = createDeviceLocalBuffer(
            device_, size_t(combNnz) * sizeof(uint32_t), "fem-hrow");
        work_.hCol = createDeviceLocalBuffer(
            device_, size_t(combNnz) * sizeof(uint32_t), "fem-hcol");
        work_.hSeg = createDeviceLocalBuffer(
            device_, size_t(combNnz + 1) * sizeof(uint32_t), "fem-hseg");
        work_.gElastic = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-gel");
        work_.xhat = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-xhat");
        work_.diff = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-diff");
        work_.mDiff = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-mdiff");
        work_.rhs = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-rhs");
        work_.p = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-p");
        work_.tmp = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-tmp");
        work_.elemE = createDeviceLocalBuffer(
            device_, size_t(M) * sizeof(double), "fem-elemE");
        work_.scalar = createDeviceLocalBuffer(
            device_, sizeof(double), "fem-scalar");

        // ---- Contact topology (global indices) + trajectory/scratch buffers ----
        std::vector<uint32_t> triConnF(size_t(nTris_) * 3), edgeConnF(size_t(nEdges_) * 2),
                              vertConnF(N);
        for (uint32_t t = 0; t < nTris_; ++t)
            for (int j = 0; j < 3; ++j) triConnF[size_t(t) * 3 + j] = uint32_t(tris_[t][j]);
        for (uint32_t e = 0; e < nEdges_; ++e)
            for (int j = 0; j < 2; ++j) edgeConnF[size_t(e) * 2 + j] = uint32_t(edges_[e][j]);
        for (uint32_t i = 0; i < N; ++i) vertConnF[i] = i;

        contact_.triConn = createDeviceLocalBuffer(
            device_, std::max<size_t>(triConnF.size(), 1) * sizeof(uint32_t),
            "fem-triconn");
        contact_.edgeConn = createDeviceLocalBuffer(
            device_, std::max<size_t>(edgeConnF.size(), 1) * sizeof(uint32_t),
            "fem-edgeconn");
        contact_.vertConn = createDeviceLocalBuffer(
            device_, size_t(N) * sizeof(uint32_t), "fem-vertconn");
        if (!triConnF.empty()) uploadBytes(contact_.triConn, triConnF.data(), triConnF.size() * sizeof(uint32_t));
        if (!edgeConnF.empty()) uploadBytes(contact_.edgeConn, edgeConnF.data(), edgeConnF.size() * sizeof(uint32_t));
        uploadBytes(contact_.vertConn, vertConnF.data(), vertConnF.size() * sizeof(uint32_t));

        contact_.vertLo = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-vlo");
        contact_.vertHi = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-vhi");
        contact_.triLo = createDeviceLocalBuffer(
            device_, std::max<size_t>(size_t(nTris_) * 3, 1) * sizeof(double),
            "fem-tlo");
        contact_.triHi = createDeviceLocalBuffer(
            device_, std::max<size_t>(size_t(nTris_) * 3, 1) * sizeof(double),
            "fem-thi");
        contact_.edgeLo = createDeviceLocalBuffer(
            device_, std::max<size_t>(size_t(nEdges_) * 3, 1) * sizeof(double),
            "fem-elo");
        contact_.edgeHi = createDeviceLocalBuffer(
            device_, std::max<size_t>(size_t(nEdges_) * 3, 1) * sizeof(double),
            "fem-ehi");
        contact_.stepDir = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-stepdir");

        // scalar/param buffers (constant)
        contact_.alpha1 = createDeviceLocalBuffer(
            device_, sizeof(double), "fem-alpha1");
        contact_.dHat = createDeviceLocalBuffer(
            device_, sizeof(double), "fem-dhat");
        contact_.dHatSqr = createDeviceLocalBuffer(
            device_, sizeof(double), "fem-dhatsqr");
        contact_.bParams = createDeviceLocalBuffer(
            device_, 2 * sizeof(double), "fem-bparams");
        contact_.typeOff = createDeviceLocalBuffer(
            device_, 5 * sizeof(uint32_t), "fem-typeoff");
        const double one = 1.0, dh = dHat_, dh2 = dHat_ * dHat_;
        const double bp[2] = {kappa_, dHat_};
        uploadBytes(contact_.alpha1, &one, sizeof(double));
        uploadBytes(contact_.dHat, &dh, sizeof(double));
        uploadBytes(contact_.dHatSqr, &dh2, sizeof(double));
        uploadBytes(contact_.bParams, bp, sizeof(bp));
    }

    // Grow combined-H BCOO buffers to hold `needNnz` blocks (grow-only).
    void GPUFEMBackend::ensureHCap(uint32_t needNnz)
    {
        if (needNnz <= work_.hCap) return;
        work_.hCap = needNnz;
        work_.hBlocks = createDeviceLocalBuffer(
            device_, size_t(needNnz) * 9 * sizeof(double), "fem-hblk");
        work_.hRow = createDeviceLocalBuffer(
            device_, size_t(needNnz) * sizeof(uint32_t), "fem-hrow");
        work_.hCol = createDeviceLocalBuffer(
            device_, size_t(needNnz) * sizeof(uint32_t), "fem-hcol");
        work_.hSeg = createDeviceLocalBuffer(
            device_, size_t(needNnz + 1) * sizeof(uint32_t), "fem-hseg");
    }

    // Grow per-candidate contact scratch (energy[total], scaled barrier blocks).
    void GPUFEMBackend::ensureContactCap(uint32_t numCand)
    {
        if (numCand <= contact_.cap) return;
        contact_.cap = numCand;
        contact_.energy = createDeviceLocalBuffer(
            device_, size_t(numCand) * sizeof(double), "fem-benergy");
        // worst case 16 Hessian blocks per constraint (PT/EE).
        contact_.h2Blocks = createDeviceLocalBuffer(
            device_, size_t(numCand) * 16 * 9 * sizeof(double), "fem-bh2");
    }

    // ============================================================================
    // readback — download positions/velocities, compute energies (per-frame only)
    // ============================================================================
    void GPUFEMBackend::readback(FEMFrame& out)
    {
        const uint32_t N = nVerts_;
        std::vector<double> xf(size_t(N) * 3), vf(size_t(N) * 3);
        downloadBytes(pos_.cur, xf.data(), xf.size() * sizeof(double));
        downloadBytes(pos_.vel, vf.data(), vf.size() * sizeof(double));

        out.positions.resize(N);
        out.velocities.resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            out.positions[i] = glm::dvec3(xf[i * 3], xf[i * 3 + 1], xf[i * 3 + 2]);
            out.velocities[i] = glm::dvec3(vf[i * 3], vf[i * 3 + 1], vf[i * 3 + 2]);
        }

        // Elastic potential (closed-form SNH over tets).
        double pe = 0.0;
        for (const auto& t : tets_)
        {
            glm::dvec3 r0 = rest_[t[0]];
            glm::dmat3 Dm(rest_[t[1]] - r0, rest_[t[2]] - r0, rest_[t[3]] - r0);
            glm::dmat3 DmInv = glm::inverse(Dm);
            double vol = glm::determinant(Dm) / 6.0;
            glm::dvec3 c0 = out.positions[t[0]];
            glm::dmat3 Ds(out.positions[t[1]] - c0, out.positions[t[2]] - c0, out.positions[t[3]] - c0);
            glm::dmat3 F = Ds * DmInv;
            double Ic = 0.0;
            for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) Ic += F[c][r] * F[c][r];
            double J = glm::determinant(F), Jm1 = J - 1.0;
            pe += (0.5 * mu_ * (Ic - 3.0) - mu_ * Jm1 + 0.5 * lambda_ * Jm1 * Jm1) * vol;
        }
        // Kinetic 0.5 v^T M v (consistent mass).
        double ke = 0.0;
        for (uint32_t t = 0; t < nTets_; ++t)
        {
            glm::dvec3 r0 = rest_[tets_[t][0]];
            glm::dmat3 Dm(rest_[tets_[t][1]] - r0, rest_[tets_[t][2]] - r0, rest_[tets_[t][3]] - r0);
            double vol = glm::determinant(Dm) / 6.0;
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                {
                    double coeff = density_ * vol * (j == k ? 0.1 : 0.05);
                    ke += 0.5 * coeff * glm::dot(out.velocities[tets_[t][j]], out.velocities[tets_[t][k]]);
                }
        }
        out.kineticEnergy = ke;
        out.potentialEnergy = pe;
        out.totalEnergy = ke + pe;
        out.time = time_;
        out.newtonIters = lastNewtonIters_;
    }

    // ============================================================================
    // step — device-resident implicit-Euler Newton step (elastic + self-contact)
    // ============================================================================
    void GPUFEMBackend::step(Real dt)
    {
        if (!valid_) return;
        const uint32_t N = nVerts_, M = nTets_, v3 = N * 3;
        const double h = dt, h2 = dt * dt;
        using B = BarrierDesc;
        auto cc = [&](CommandList& c) { c.memoryBarrier(B::StageComputeShader, B::StageComputeShader); };

        // TODO: should be extracted to RHI
        auto dispatch1 = [&](const PipelineRef& pso, auto& p, uint32_t n, uint32_t wg = kWG)
        {
            auto cmd = device_.beginCommands(QueueType::Compute);
            cmd->dispatch(pso, p, (n + wg - 1) / wg, 1, 1);
            device_.submitAndWait(*cmd, QueueType::Compute);
        };
        auto axpy = [&](const BufferRef& y, const BufferRef& x, double a, uint32_t n)
        {
            AxpyP p;
            p.y = y;
            p.x = x;
            p.a = a;
            p.n = n;
            dispatch1(psoAxpy_, p, n);
        };
        auto fill0 = [&](const BufferRef& b, size_t bytes)
        {
            auto cmd = device_.beginCommands(QueueType::Compute);
            cmd->fillBuffer(b, 0u);
            device_.submitAndWait(*cmd, QueueType::Compute);
        };
        // TODO: perhaps copy vector can be extracted.
        // TODO: consider whether vector on GPU can be abstracted as a new class rather than BufferRef
        auto copyVec = [&](const BufferRef& src, const BufferRef& dst)
        {
            auto cmd = device_.beginCommands(QueueType::Transfer);
            std::array<BufferCopy, 1> rg{{{0, 0, size_t(v3) * sizeof(double)}}};
            cmd->copyBuffer(src, dst, rg);
            device_.submitAndWait(*cmd, QueueType::Transfer);
        };
        // Device buffer copy with byte offsets (for BCOO concatenation).
        auto copyBytes = [&](const BufferRef& src, const BufferRef& dst,
                             size_t srcOff, size_t dstOff, size_t bytes)
        {
            if (bytes == 0) return;
            auto cmd = device_.beginCommands(QueueType::Transfer);
            std::array<BufferCopy, 1> rg{{{srcOff, dstOff, bytes}}};
            cmd->copyBuffer(src, dst, rg);
            device_.submitAndWait(*cmd, QueueType::Transfer);
        };
        auto elasticGrad = [&](const BufferRef& xbuf, const BufferRef& gout)
        {
            EGradP p;
            p.grad = gout;
            p.x = xbuf;
            p.tetConn = tet_.conn;
            p.DmInv = tet_.dmInv;
            p.vol = tet_.vol;
            p.adjStart = tet_.adjStart;
            p.adjTet = tet_.adjTet;
            p.adjLocal = tet_.adjLocal;
            p.material = tet_.material;
            p.numVerts = N;
            dispatch1(psoEGrad_, p, N);
        };
        auto massApply = [&](const BufferRef& vin, const BufferRef& yout)
        {
            MassP p;
            p.y = yout;
            p.v = vin;
            p.tetConn = tet_.conn;
            p.vol = tet_.vol;
            p.adjStart = tet_.adjStart;
            p.adjTet = tet_.adjTet;
            p.adjLocal = tet_.adjLocal;
            p.density = density_;
            p.numVerts = N;
            dispatch1(psoMass_, p, N);
        };
        // Total IP energy at `xbuf`: inertial + h^2 * (elastic + barrier). The
        // barrier term sums per-constraint energies over the `total` active
        // constraints (activation_->pairs() + contact_.typeOff, current iteration);
        // pass total==0 to skip it.
        auto energyAt = [&](const BufferRef& xbuf, uint32_t total)-> double
        {
            // elastic
            {
                EEnergyP p;
                p.energyOut = work_.elemE;
                p.x = xbuf;
                p.tetConn = tet_.conn;
                p.DmInv = tet_.dmInv;
                p.vol = tet_.vol;
                p.material = tet_.material;
                p.numTets = M;
                auto cmd = device_.beginCommands(QueueType::Compute);
                cmd->dispatch(psoEEnergy_, p, groups(M), 1, 1);
                cc(*cmd);
                reduce_->run(*cmd, ksk::rpk::ReduceOp::Sum, ksk::rpk::ScalarType::Float64,
                             work_.elemE, work_.scalar, M);
                device_.submitAndWait(*cmd, QueueType::Compute);
            }
            double Eel = readbackScalar(work_.scalar);
            // barrier (per active constraint -> Sum)
            double Ebar = 0.0;
            if (total > 0)
            {
                BEnergyP p;
                p.energyOut = contact_.energy;
                p.pairs = activation_->pairs();
                p.x = xbuf;
                p.xRest = pos_.rest;
                p.typeOff = contact_.typeOff;
                p.dParams = contact_.bParams;
                p.total = total;
                auto cmd = device_.beginCommands(QueueType::Compute);
                cmd->dispatch(psoBEnergy_, p, groups(total), 1, 1);
                cc(*cmd);
                reduce_->run(*cmd, ksk::rpk::ReduceOp::Sum, ksk::rpk::ScalarType::Float64,
                             contact_.energy, work_.scalar, total);
                device_.submitAndWait(*cmd, QueueType::Compute);
                Ebar = readbackScalar(work_.scalar);
            }
            // inertial 0.5 (x-xhat)^T M (x-xhat)
            copyVec(xbuf, work_.diff);
            axpy(work_.diff, work_.xhat, -1.0, v3);
            massApply(work_.diff, work_.mDiff);
            {
                MulP p;
                p.outv = work_.tmp;
                p.a = work_.diff;
                p.b = work_.mDiff;
                p.n = v3;
                auto cmd = device_.beginCommands(QueueType::Compute);
                cmd->dispatch(psoMul_, p, groups(v3), 1, 1);
                cc(*cmd);
                reduce_->run(*cmd, ksk::rpk::ReduceOp::Sum, ksk::rpk::ScalarType::Float64,
                             work_.tmp, work_.scalar, v3);
                device_.submitAndWait(*cmd, QueueType::Compute);
            }
            double inertial = 0.5 * readbackScalar(work_.scalar);
            return inertial + h2 * (Eel + Ebar);
        };

        // xhat = x_t + h*v + h2*g, the inertial target for this step.
        copyVec(pos_.cur, work_.xhat); // xhat = x_t
        axpy(work_.xhat, pos_.vel, h, v3); // += h v
        // += h2 * g : gravity is uniform; stage it into work_.tmp then axpy.
        {
            std::vector<double> gflat(size_t(v3), 0.0);
            // TODO: that's not good, do gravity and other potential energies(exclude elastic) on GPU
            for (uint32_t i = 0; i < N; ++i)
            {
                gflat[i * 3] = gravity_.x;
                gflat[i * 3 + 1] = gravity_.y;
                gflat[i * 3 + 2] = gravity_.z;
            }
            uploadBytes(work_.tmp, gflat.data(), gflat.size() * sizeof(double));
            axpy(work_.xhat, work_.tmp, h2, v3); // xhat += h2 * g
        }

        // Keep x_t (= current pos_.cur) for the velocity update.
        auto dXt = createDeviceLocalBuffer(
            device_, size_t(v3) * sizeof(double), "fem-xt");
        copyVec(pos_.cur, dXt);

        // ---- broad phase (once per step): deformable self-contact VT/EE candidates
        //      over the predicted trajectory x_t -> x_hat; reused by activation/CCD
        //      across the Newton iterations. ----
        copyVec(work_.xhat, contact_.stepDir);
        axpy(contact_.stepDir, dXt, -1.0, v3); // stepDir = xhat - x_t
        uint32_t numVt = 0, numEe = 0;
        if (nTris_ >= 2)
        {
            //TODO: we had better use a struct to pass these arguments
            traj_->compute(pos_.cur, contact_.stepDir, contact_.vertConn, contact_.alpha1, contact_.vertLo,
                           contact_.vertHi, N, 1);
            traj_->compute(pos_.cur, contact_.stepDir, contact_.triConn, contact_.alpha1, contact_.triLo,
                           contact_.triHi, nTris_, 3);
            triBvh_->build(contact_.triLo, contact_.triHi, nTris_);
            numVt = broad_->queryVT(*triBvh_, contact_.vertLo, contact_.vertHi, contact_.triConn, contact_.dHat, N);
        }
        if (nEdges_ >= 2)
        {
            traj_->compute(pos_.cur, contact_.stepDir, contact_.edgeConn, contact_.alpha1, contact_.edgeLo,
                           contact_.edgeHi, nEdges_, 2);
            edgeBvh_->build(contact_.edgeLo, contact_.edgeHi, nEdges_);
            numEe = broad_->queryEE(*edgeBvh_, contact_.edgeLo, contact_.edgeHi, contact_.edgeConn, contact_.dHat,
                                    nEdges_);
        }
        const uint32_t totalCand = numVt + numEe;
        if (totalCand > 0) ensureContactCap(totalCand);

        const int maxNewton = 50;
        int iter = 0;
        for (; iter < maxNewton; ++iter)
        {
            // --- activation: classify candidates at current x -> active PP/PE/PT/EE ---
            // TODO: support collision detection against scene colliders
            gpu_ns::GpuActivation::Result act;
            gpu_ns::GpuBarrierAssembler::Result bres;
            uint32_t total = 0;
            if (totalCand > 0)
            {
                act = activation_->activate(pos_.cur, broad_->vtPairs(), numVt,
                                            broad_->eePairs(), numEe, contact_.dHatSqr);
                total = act.numConstraints;
                if (total > 0)
                {
                    bres = barrier_->assemble(pos_.cur, activation_->pairs(), act.typeOffsets,
                                              total, kappa_, dHat_, pos_.rest);
                    uploadBytes(contact_.typeOff, act.typeOffsets.data(), 5 * sizeof(uint32_t));
                }
            }
            const uint32_t Hn = bres.numHessBlocks, Gn = bres.numGradEntries;

            // --- assemble elastic Hessian at current x ---
            {
                EHessP p;
                p.outBlocks = work_.ehBlocks;
                p.outRow = work_.ehRow;
                p.outCol = work_.ehCol;
                p.x = pos_.cur;
                p.tetConn = tet_.conn;
                p.DmInv = tet_.dmInv;
                p.vol = tet_.vol;
                p.material = tet_.material;
                p.numTets = M;
                dispatch1(psoEHess_, p, M, 64);
            }

            // --- combined H = h^2*(elastic + barrier) ++ mass (BCOO concat) ---
            const uint32_t ehN = M * 16, ehDoubles = ehN * 9;
            const uint32_t combN = ehN + Hn + mass_.nnz;
            ensureHCap(combN);
            fill0(work_.hBlocks, 0); // zero whole buffer
            axpy(work_.hBlocks, work_.ehBlocks, h2, ehDoubles); // [0..ehN) = h^2 * elastic
            // TODO: so many copyBytes calls, that's insane
            copyBytes(work_.ehRow, work_.hRow, 0, 0, size_t(ehN) * sizeof(uint32_t));
            copyBytes(work_.ehCol, work_.hCol, 0, 0, size_t(ehN) * sizeof(uint32_t));
            if (Hn > 0)
            {
                // [ehN..ehN+Hn) = h^2 * barrier
                fill0(contact_.h2Blocks, 0);
                axpy(contact_.h2Blocks, barrier_->hessBlocks(), h2, Hn * 9);
                copyBytes(contact_.h2Blocks, work_.hBlocks, 0, size_t(ehDoubles) * sizeof(double),
                          size_t(Hn) * 9 * sizeof(double));
                copyBytes(barrier_->hessRow(), work_.hRow, 0, size_t(ehN) * sizeof(uint32_t),
                          size_t(Hn) * sizeof(uint32_t));
                copyBytes(barrier_->hessCol(), work_.hCol, 0, size_t(ehN) * sizeof(uint32_t),
                          size_t(Hn) * sizeof(uint32_t));
            }
            const uint32_t massOff = ehN + Hn; // [massOff..] = mass
            copyBytes(mass_.blocks, work_.hBlocks, 0, size_t(massOff) * 9 * sizeof(double),
                      size_t(mass_.nnz) * 9 * sizeof(double));
            copyBytes(mass_.row, work_.hRow, 0, size_t(massOff) * sizeof(uint32_t),
                      size_t(mass_.nnz) * sizeof(uint32_t));
            copyBytes(mass_.col, work_.hCol, 0, size_t(massOff) * sizeof(uint32_t),
                      size_t(mass_.nnz) * sizeof(uint32_t));

            // --- RHS: negG = -( M(x-xhat) + h^2*(gElastic + gBarrier) ) ---
            elasticGrad(pos_.cur, work_.gElastic); // gElastic
            if (Gn > 0) // += gBarrier (scatter-add)
                gradReduce_->addInto(work_.gElastic, barrier_->gradRow(), barrier_->gradVal(), Gn);
            copyVec(pos_.cur, work_.diff);
            axpy(work_.diff, work_.xhat, -1.0, v3); // diff = x - xhat
            massApply(work_.diff, work_.mDiff); // Mdiff
            fill0(work_.rhs, 0);
            axpy(work_.rhs, work_.mDiff, -1.0, v3);
            axpy(work_.rhs, work_.gElastic, -h2, v3);

            // --- sort + solve ---
            uint32_t numSeg = sorter_->sort(work_.hBlocks, work_.hRow, work_.hCol, work_.hSeg, combN);
            if (numSeg == 0) break;
            fill0(work_.p, 0);
            pcg_->solveDevice(work_.hBlocks, work_.hRow, work_.hCol, work_.hSeg, N, combN, numSeg,
                              work_.rhs, work_.p, pcgMaxIter_, pcgTol_);

            // --- convergence: ||p||_inf < eps * lengthScale * h ---
            {
                AbsP p;
                p.outv = work_.tmp;
                p.inv = work_.p;
                p.n = v3;
                auto cmd = device_.beginCommands(QueueType::Compute);
                cmd->dispatch(psoAbs_, p, groups(v3), 1, 1);
                cc(*cmd);
                reduce_->run(*cmd, ksk::rpk::ReduceOp::Max, ksk::rpk::ScalarType::Float64,
                             work_.tmp, work_.scalar, v3);
                device_.submitAndWait(*cmd, QueueType::Compute);
            }
            double pInf = readbackScalar(work_.scalar);
            if (pInf < eps_ * lengthScale_ * h) break;

            // --- CCD: conservative step-size upper bound over the candidates ---
            double alphaMax = 1.0;
            if (totalCand > 0)
                alphaMax = ccd_->stepSizeUpperBound(pos_.cur, work_.p, broad_->vtPairs(), numVt,
                                                    broad_->eePairs(), numEe, 1.0, 0.1);

            // --- backtracking line search on the IP energy (elastic+barrier+inertial) ---
            double Eprev = energyAt(pos_.cur, total);
            double alpha = (alphaMax < 1.0) ? alphaMax : 1.0;
            int ls = 0;
            for (; ls < 32; ++ls)
            {
                copyVec(pos_.cur, work_.rhs); // trial in work_.rhs (free after solve)
                axpy(work_.rhs, work_.p, alpha, v3); // trial = x + alpha p
                double E = energyAt(work_.rhs, total);
                if (E <= Eprev) break;
                alpha *= 0.5;
            }
            // commit: x += alpha p
            axpy(pos_.cur, work_.p, alpha, v3);
        }
        lastNewtonIters_ = iter;

        // velocity update: v = (x - x_t) / h
        fill0(pos_.vel, size_t(v3) * sizeof(double));
        axpy(pos_.vel, pos_.cur, 1.0 / h, v3);
        axpy(pos_.vel, dXt, -1.0 / h, v3);

        time_ += dt;
    }

    // ============================================================================
    // Factory
    // ============================================================================
    // TODO:  this shouldn't be placed in gpu-fem-backend.cc
    std::unique_ptr<FEMBackend> createFEMBackend(const std::string& type,
                                                 ksk::rhi::Device& device,
                                                 ksk::rhi::ShaderCompiler& compiler)
    {
        if (type == "gpu") return std::make_unique<GPUFEMBackend>(device, compiler);
        if (type == "cpu") return std::make_unique<CPUFEMBackend>();
        throw std::runtime_error("[createFEMBackend] Unknown backend type: " + type);
    }
} // namespace ksk::fem
