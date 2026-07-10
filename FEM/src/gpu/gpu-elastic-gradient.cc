// ============================================================================
// FEM/src/gpu/gpu-elastic-gradient.cc
// ============================================================================
#include <fem/gpu/gpu-elastic-gradient.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <span>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace ksk::fem::gpu {

using namespace ksk::rhi;
namespace fs = std::filesystem;

GpuElasticGradient::GpuElasticGradient(Device& device, ShaderCompiler& compiler,
                                       const fs::path& shaderDir)
    : device_(device) {
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;
    pso_ = compileComputePipeline(device, compiler, dir / "elastic-gradient.hlsl");
    valid_ = pso_.valid();
    if (!valid_) spdlog::error("[GpuElasticGradient] pipeline compilation failed");
}

void GpuElasticGradient::uploadBytes(const BufferRef& dst, const void* data, size_t bytes) {
    if (bytes == 0) return;
    auto staging = device_.createBuffer({
        .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
        .usage = BufferDesc::TransferSrc});
    std::memcpy(staging->map(), data, bytes);
    staging->unmap();
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(staging, dst, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
}

void GpuElasticGradient::download(const BufferRef& src, void* dst, size_t bytes) {
    auto rb = device_.createBuffer({
        .sizeBytes = bytes, .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst});
    auto cmd = device_.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(src, rb, region);
    device_.submitAndWait(*cmd, QueueType::Transfer);
    std::memcpy(dst, rb->map(), bytes);
    rb->unmap();
}

// ---- SHADER_PARAMS ----
SHADER_PARAMS_BEGIN(ElasticGradParams)
    SHADER_PARAM_UAV   (BufferRef, grad);
    SHADER_PARAM_SRV   (BufferRef, x);
    SHADER_PARAM_SRV   (BufferRef, tetConn);
    SHADER_PARAM_SRV   (BufferRef, DmInv);
    SHADER_PARAM_SRV   (BufferRef, vol);
    SHADER_PARAM_SRV   (BufferRef, adjStart);
    SHADER_PARAM_SRV   (BufferRef, adjTet);
    SHADER_PARAM_SRV   (BufferRef, adjLocal);
    SHADER_PARAM_SRV   (BufferRef, material);
    SHADER_PARAM_SCALAR(uint32_t,  numVerts);
SHADER_PARAMS_END();

void GpuElasticGradient::compute(const std::vector<glm::dvec3>& restVerts,
                                 const std::vector<glm::dvec3>& curVerts,
                                 const std::vector<std::array<int, 4>>& tets,
                                 double mu, double lambda,
                                 std::vector<glm::dvec3>& gradOut) {
    const uint32_t N = static_cast<uint32_t>(restVerts.size());
    const uint32_t M = static_cast<uint32_t>(tets.size());
    gradOut.assign(N, glm::dvec3(0.0));
    if (!valid_ || N == 0 || M == 0) return;

    // ---- CPU precompute: DmInv (column-major) + volume per tet ----
    std::vector<glm::dmat3> dmInv(M);
    std::vector<double>   vol(M);
    std::vector<uint32_t> conn(size_t(M) * 4);
    for (uint32_t t = 0; t < M; ++t) {
        const auto& tet = tets[t];
        glm::dvec3 p0 = restVerts[tet[0]];
        glm::dmat3 Dm(restVerts[tet[1]] - p0,   // column 0
                      restVerts[tet[2]] - p0,   // column 1
                      restVerts[tet[3]] - p0);  // column 2
        dmInv[t] = glm::inverse(Dm);
        vol[t] = glm::determinant(Dm) / 6.0;
        for (int j = 0; j < 4; ++j) conn[size_t(t) * 4 + j] = static_cast<uint32_t>(tet[j]);
    }

    // ---- vertex -> incident (tet, localIndex) adjacency (CSR-like) ----
    std::vector<uint32_t> adjStart(N + 1, 0);
    for (uint32_t t = 0; t < M; ++t)
        for (int j = 0; j < 4; ++j) adjStart[tets[t][j] + 1]++;
    for (uint32_t i = 0; i < N; ++i) adjStart[i + 1] += adjStart[i];
    const uint32_t K = adjStart[N];
    std::vector<uint32_t> adjTet(K), adjLocal(K);
    std::vector<uint32_t> cursor(adjStart.begin(), adjStart.end());
    for (uint32_t t = 0; t < M; ++t)
        for (int j = 0; j < 4; ++j) {
            uint32_t v = static_cast<uint32_t>(tets[t][j]);
            uint32_t slot = cursor[v]++;
            adjTet[slot] = t;
            adjLocal[slot] = static_cast<uint32_t>(j);
        }

    const double material[2] = {mu, lambda};

    // ---- buffers ----
    auto bGrad = createDeviceLocalBuffer(
        device_, size_t(N) * 3 * sizeof(double), "eg-grad");
    auto bX = createDeviceLocalBuffer(
        device_, size_t(N) * 3 * sizeof(double), "eg-x");
    auto bConn = createDeviceLocalBuffer(
        device_, size_t(M) * 4 * sizeof(uint32_t), "eg-conn");
    auto bDmInv = createDeviceLocalBuffer(
        device_, size_t(M) * 9 * sizeof(double), "eg-dminv");
    auto bVol = createDeviceLocalBuffer(
        device_, size_t(M) * sizeof(double), "eg-vol");
    auto bAdjStart = createDeviceLocalBuffer(
        device_, size_t(N + 1) * sizeof(uint32_t), "eg-adjStart");
    auto bAdjTet = createDeviceLocalBuffer(
        device_, size_t(K) * sizeof(uint32_t), "eg-adjTet");
    auto bAdjLocal = createDeviceLocalBuffer(
        device_, size_t(K) * sizeof(uint32_t), "eg-adjLocal");
    auto bMat = createDeviceLocalBuffer(
        device_, 2 * sizeof(double), "eg-material");

    auto xBytes = asStructuredBytes(std::span<const glm::dvec3>(curVerts));
    auto dmInvBytes = asStructuredBytes(std::span<const glm::dmat3>(dmInv));
    uploadBytes(bX, xBytes.data(), xBytes.size());
    uploadBytes(bConn, conn.data(), conn.size() * sizeof(uint32_t));
    uploadBytes(bDmInv, dmInvBytes.data(), dmInvBytes.size());
    uploadBytes(bVol, vol.data(), vol.size() * sizeof(double));
    uploadBytes(bAdjStart, adjStart.data(), adjStart.size() * sizeof(uint32_t));
    uploadBytes(bAdjTet, adjTet.data(), adjTet.size() * sizeof(uint32_t));
    uploadBytes(bAdjLocal, adjLocal.data(), adjLocal.size() * sizeof(uint32_t));
    uploadBytes(bMat, material, sizeof(material));

    // ---- dispatch ----
    auto cmd = device_.beginCommands(QueueType::Compute);
    cmd->memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                       BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderRead);
    ElasticGradParams p;
    p.grad = bGrad; p.x = bX; p.tetConn = bConn; p.DmInv = bDmInv; p.vol = bVol;
    p.adjStart = bAdjStart; p.adjTet = bAdjTet; p.adjLocal = bAdjLocal;
    p.material = bMat; p.numVerts = N;
    cmd->dispatch(pso_, p, (N + 255) / 256, 1, 1);
    cmd->memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageTransfer,
                       BarrierDesc::AccessShaderWrite, BarrierDesc::AccessTransferRead);
    device_.submitAndWait(*cmd, QueueType::Compute);

    // ---- download ----
    auto gradBytes = asWritableStructuredBytes(std::span<glm::dvec3>(gradOut));
    download(bGrad, gradBytes.data(), gradBytes.size());
}

} // namespace ksk::fem::gpu
