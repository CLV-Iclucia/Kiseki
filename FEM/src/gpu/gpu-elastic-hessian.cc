// ============================================================================
// FEM/src/gpu/gpu-elastic-hessian.cc
// ============================================================================
#include <fem/gpu/gpu-elastic-hessian.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <span>
#include <string>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace sim::fem::gpu {

using namespace sim::rhi;
namespace fs = std::filesystem;

static PipelineRef compileWithDefines(
    Device& device, ShaderCompiler& compiler, const fs::path& path,
    const std::vector<std::pair<std::string, std::string>>& defines) {
    ShaderCompileOptions opts;
    opts.entryPoint    = "main";
    opts.defines       = defines;
    return compileComputePipeline(device, compiler, path, opts);
}

GpuElasticHessian::GpuElasticHessian(Device& device, ShaderCompiler& compiler,
                                     const fs::path& shaderDir, int cyclicSweeps)
    : device_(device) {
    fs::path dir = shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir;
    // Cyclic + abs clamp (matches CPU filteredEnergyHessian); sweeps tunable.
    std::vector<std::pair<std::string, std::string>> defs = {
        {"JACOBI_SWEEPS", std::to_string(cyclicSweeps)},
    };
    pso_ = compileWithDefines(device, compiler, dir / "elastic-hessian.hlsl", defs);
    valid_ = pso_.valid();
    if (!valid_) spdlog::error("[GpuElasticHessian] pipeline compilation failed");
}

void GpuElasticHessian::uploadBytes(const BufferRef& dst, const void* data, size_t bytes) {
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

void GpuElasticHessian::download(const BufferRef& src, void* dst, size_t bytes) {
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

SHADER_PARAMS_BEGIN(ElasticHessParams)
    SHADER_PARAM_UAV   (BufferRef, outBlocks);
    SHADER_PARAM_UAV   (BufferRef, outRow);
    SHADER_PARAM_UAV   (BufferRef, outCol);
    SHADER_PARAM_SRV   (BufferRef, x);
    SHADER_PARAM_SRV   (BufferRef, tetConn);
    SHADER_PARAM_SRV   (BufferRef, DmInv);
    SHADER_PARAM_SRV   (BufferRef, vol);
    SHADER_PARAM_SRV   (BufferRef, material);
    SHADER_PARAM_SCALAR(uint32_t,  numTets);
SHADER_PARAMS_END();

void GpuElasticHessian::compute(const std::vector<glm::dvec3>& restVerts,
                                const std::vector<glm::dvec3>& curVerts,
                                const std::vector<std::array<int, 4>>& tets,
                                double mu, double lambda,
                                std::vector<double>& blocksOut,
                                std::vector<uint32_t>& rowOut,
                                std::vector<uint32_t>& colOut) {
    const uint32_t E = static_cast<uint32_t>(tets.size()) * 16;
    blocksOut.assign(size_t(E) * 9, 0.0);
    rowOut.assign(E, 0);
    colOut.assign(E, 0);

    DeviceBcoo m = computeToDevice(restVerts, curVerts, tets, mu, lambda);
    if (m.nnz == 0) return;

    download(m.blocks, blocksOut.data(), blocksOut.size() * sizeof(double));
    download(m.row, rowOut.data(), rowOut.size() * sizeof(uint32_t));
    download(m.col, colOut.data(), colOut.size() * sizeof(uint32_t));
}

DeviceBcoo GpuElasticHessian::computeToDevice(
        const std::vector<glm::dvec3>& restVerts,
        const std::vector<glm::dvec3>& curVerts,
        const std::vector<std::array<int, 4>>& tets,
        double mu, double lambda) {
    DeviceBcoo out;
    const uint32_t N = static_cast<uint32_t>(restVerts.size());
    const uint32_t M = static_cast<uint32_t>(tets.size());
    const uint32_t E = M * 16;  // entries
    out.nVerts = N;
    if (!valid_ || N == 0 || M == 0) return out;

    // CPU precompute DmInv (column-major) + volume + connectivity
    std::vector<glm::dmat3> dmInv(M);
    std::vector<double>   vol(M);
    std::vector<uint32_t> conn(size_t(M) * 4);
    for (uint32_t t = 0; t < M; ++t) {
        const auto& tet = tets[t];
        glm::dvec3 p0 = restVerts[tet[0]];
        glm::dmat3 Dm(restVerts[tet[1]] - p0, restVerts[tet[2]] - p0, restVerts[tet[3]] - p0);
        dmInv[t] = glm::inverse(Dm);
        vol[t] = glm::determinant(Dm) / 6.0;
        for (int j = 0; j < 4; ++j) conn[size_t(t) * 4 + j] = static_cast<uint32_t>(tet[j]);
    }
    const double material[2] = {mu, lambda};

    // (Re)allocate persistent output buffers on growth.
    if (E > capEntries_) {
        capEntries_ = E;
        bBlocks_ = createDeviceLocalBuffer(
            device_, size_t(E) * 9 * sizeof(double), "eh-blocks");
        bRow_ = createDeviceLocalBuffer(
            device_, size_t(E) * sizeof(uint32_t), "eh-row");
        bCol_ = createDeviceLocalBuffer(
            device_, size_t(E) * sizeof(uint32_t), "eh-col");
    }

    auto bX = createDeviceLocalBuffer(
        device_, size_t(N) * 3 * sizeof(double), "eh-x");
    auto bConn = createDeviceLocalBuffer(
        device_, size_t(M) * 4 * sizeof(uint32_t), "eh-conn");
    auto bDmInv = createDeviceLocalBuffer(
        device_, size_t(M) * 9 * sizeof(double), "eh-dminv");
    auto bVol = createDeviceLocalBuffer(
        device_, size_t(M) * sizeof(double), "eh-vol");
    auto bMat = createDeviceLocalBuffer(
        device_, 2 * sizeof(double), "eh-material");

    auto xBytes = asStructuredBytes(std::span<const glm::dvec3>(curVerts));
    auto dmInvBytes = asStructuredBytes(std::span<const glm::dmat3>(dmInv));
    uploadBytes(bX, xBytes.data(), xBytes.size());
    uploadBytes(bConn, conn.data(), conn.size() * sizeof(uint32_t));
    uploadBytes(bDmInv, dmInvBytes.data(), dmInvBytes.size());
    uploadBytes(bVol, vol.data(), vol.size() * sizeof(double));
    uploadBytes(bMat, material, sizeof(material));

    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        cmd->memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                           BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderRead);
        ElasticHessParams p;
        p.outBlocks = bBlocks_; p.outRow = bRow_; p.outCol = bCol_;
        p.x = bX; p.tetConn = bConn; p.DmInv = bDmInv; p.vol = bVol; p.material = bMat;
        p.numTets = M;
        cmd->dispatch(pso_, p, (M + 63) / 64, 1, 1);
        cmd->memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageTransfer,
                           BarrierDesc::AccessShaderWrite, BarrierDesc::AccessTransferRead);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    out.blocks = bBlocks_;
    out.row    = bRow_;
    out.col    = bCol_;
    out.nnz    = E;
    return out;
}

} // namespace sim::fem::gpu
