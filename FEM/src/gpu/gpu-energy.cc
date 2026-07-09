// ============================================================================
// FEM/src/gpu/gpu-energy.cc
// GPU barrier-augmented incremental potential energy (elastic term first).
// ============================================================================
#include <fem/gpu/gpu-energy.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <span>
#include <vector>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace sim::fem::gpu
{
    using namespace sim::rhi;
    namespace fs = std::filesystem;

    static constexpr uint32_t kWG = 256;
    static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

    // ---- SHADER_PARAMS (must match energy-elastic.hlsl binding names) ----
    SHADER_PARAMS_BEGIN(EnergyElasticParams)
        SHADER_PARAM_UAV(BufferRef, energyOut);
        SHADER_PARAM_SRV(BufferRef, x);
        SHADER_PARAM_SRV(BufferRef, tetConn);
        SHADER_PARAM_SRV(BufferRef, DmInv);
        SHADER_PARAM_SRV(BufferRef, vol);
        SHADER_PARAM_SRV(BufferRef, material);
        SHADER_PARAM_SCALAR(uint32_t, numTets);
    SHADER_PARAMS_END();

    GPUEnergy::GPUEnergy(Device& device, ShaderCompiler& compiler, const fs::path& shaderDir)
        : device_(device)
    {
        fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;
        reduce_ = std::make_unique<sim::rpk::Reduce>(device, compiler);
        psoElastic_ = compileComputePipeline(device, compiler, dir / "energy-elastic.hlsl");
        valid_ = psoElastic_.valid();
        if (!valid_) spdlog::error("[GpuEnergy] pipeline compilation failed");
    }

    void GPUEnergy::uploadBytes(const BufferRef& dst, const void* data, size_t bytes)
    {
        if (bytes == 0) return;
        auto staging = device_.createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
            .usage = BufferDesc::TransferSrc
        });
        std::memcpy(staging->map(), data, bytes);
        staging->unmap();
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
        cmd->copyBuffer(staging, dst, region);
        device_.submitAndWait(*cmd, QueueType::Transfer);
    }

    double GPUEnergy::readbackDouble(const BufferRef& src)
    {
        auto rb = device_.createBuffer({
            .sizeBytes = sizeof(double), .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst
        });
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> region{{{0, 0, sizeof(double)}}};
        cmd->copyBuffer(src, rb, region);
        device_.submitAndWait(*cmd, QueueType::Transfer);
        double v = *static_cast<double*>(rb->map());
        rb->unmap();
        return v;
    }

    double GPUEnergy::elastic(const std::vector<glm::dvec3>& restVerts,
                              const std::vector<glm::dvec3>& curVerts,
                              const std::vector<std::array<int, 4>>& tets,
                              double mu, double lambda)
    {
        const uint32_t N = static_cast<uint32_t>(restVerts.size());
        const uint32_t M = static_cast<uint32_t>(tets.size());
        if (!valid_ || N == 0 || M == 0) return 0.0;

        // ---- CPU precompute: DmInv (column-major) + rest volume + connectivity ----
        std::vector<glm::dmat3> dmInv(M);
        std::vector<double> vol(M);
        std::vector<uint32_t> conn(size_t(M) * 4);
        // TODO: this should be done on GPU
        for (uint32_t t = 0; t < M; ++t)
        {
            const auto& tet = tets[t];
            glm::dvec3 p0 = restVerts[tet[0]];
            glm::dmat3 Dm(restVerts[tet[1]] - p0, restVerts[tet[2]] - p0, restVerts[tet[3]] - p0);
            dmInv[t] = glm::inverse(Dm);
            vol[t] = glm::determinant(Dm) / 6.0;
            for (int j = 0; j < 4; ++j) conn[size_t(t) * 4 + j] = static_cast<uint32_t>(tet[j]);
        }

        const double material[2] = {mu, lambda};

        // ---- (re)allocate device scratch ----
        if (M > capTets_)
        {
            capTets_ = M;
            bElemE_ = createDeviceLocalBuffer(
                device_, size_t(M) * sizeof(double), "en-elem");
            bConn_ = createDeviceLocalBuffer(
                device_, size_t(M) * 4 * sizeof(uint32_t), "en-conn");
            bDmInv_ = createDeviceLocalBuffer(
                device_, size_t(M) * 9 * sizeof(double), "en-dminv");
            bVol_ = createDeviceLocalBuffer(
                device_, size_t(M) * sizeof(double), "en-vol");
        }
        if (N > capVerts_)
        {
            capVerts_ = N;
            bX_ = createDeviceLocalBuffer(
                device_, size_t(N) * 3 * sizeof(double), "en-x");
        }
        if (!bMat_)
            bMat_ = createDeviceLocalBuffer(
                device_, 2 * sizeof(double), "en-material");
        if (!bResult_)
            bResult_ = createDeviceLocalBuffer(
                device_, sizeof(double), "en-result");

        auto xBytes = asStructuredBytes(std::span<const glm::dvec3>(curVerts));
        auto dmInvBytes = asStructuredBytes(std::span<const glm::dmat3>(dmInv));
        uploadBytes(bX_, xBytes.data(), xBytes.size());
        uploadBytes(bConn_, conn.data(), conn.size() * sizeof(uint32_t));
        uploadBytes(bDmInv_, dmInvBytes.data(), dmInvBytes.size());
        uploadBytes(bVol_, vol.data(), vol.size() * sizeof(double));
        uploadBytes(bMat_, material, sizeof(material));

        using B = BarrierDesc;
        {
            auto cmd = device_.beginCommands(QueueType::Compute);
            cmd->memoryBarrier(B::StageTransfer, B::StageComputeShader,
                               B::AccessTransferWrite, B::AccessShaderRead);
            EnergyElasticParams p;
            p.energyOut = bElemE_;
            p.x = bX_;
            p.tetConn = bConn_;
            p.DmInv = bDmInv_;
            p.vol = bVol_;
            p.material = bMat_;
            p.numTets = M;
            cmd->dispatch(psoElastic_, p, groups(M), 1, 1);
            cmd->memoryBarrier(B::StageComputeShader, B::StageComputeShader);
            reduce_->run(*cmd, rpk::ReduceOp::Sum, rpk::ScalarType::Float64,
                         bElemE_, bResult_, M);
            device_.submitAndWait(*cmd, QueueType::Compute);
        }
        return readbackDouble(bResult_);
    }
} // namespace sim::fem::gpu
