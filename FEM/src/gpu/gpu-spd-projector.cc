// ============================================================================
// FEM/src/gpu/gpu-spd-projector.cc
// ============================================================================
#include <fem/gpu/gpu-spd-projector.h>

#include <spdlog/spdlog.h>

#include <array>
#include <cstring>
#include <string>

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

namespace ksk::fem::gpu {

using namespace ksk::rhi;
namespace fs = std::filesystem;

// Compile with custom preprocessor defines (compileComputePipeline can't).
static PipelineRef compileWithDefines(
    Device& device, ShaderCompiler& compiler, const fs::path& path,
    const std::vector<std::pair<std::string, std::string>>& defines) {
    ShaderCompileOptions opts;
    opts.entryPoint    = "main";
    opts.defines       = defines;
    return compileComputePipeline(device, compiler, path, opts);
}

GpuSpdProjector9::GpuSpdProjector9(Device& device, ShaderCompiler& compiler,
                                   const fs::path& shaderDir,
                                   int cyclicSweeps, int classicalRotations)
    : device_(device) {
    fs::path file = (shaderDir.empty() ? fs::path(FEM_SHADER_DIR) : shaderDir)
                    / "spd-project-9x9.hlsl";

    const std::string sweeps = std::to_string(cyclicSweeps);
    const std::string rots   = std::to_string(classicalRotations);

    valid_ = true;
    for (int strat = 0; strat < 2; ++strat) {
        for (int clamp = 0; clamp < 2; ++clamp) {
            std::vector<std::pair<std::string, std::string>> defs = {
                {"JACOBI_SWEEPS", sweeps},
                {"JACOBI_ROTATIONS", rots},
            };
            if (strat == static_cast<int>(Strategy::Classical))
                defs.push_back({"JACOBI_CLASSICAL", "1"});
            if (clamp == static_cast<int>(Clamp::Psd))
                defs.push_back({"CLAMP_PSD", "1"});

            pso_[strat][clamp] = compileWithDefines(device, compiler, file, defs);
            valid_ = valid_ && pso_[strat][clamp].valid();
        }
    }
    if (!valid_) spdlog::error("[GpuSpdProjector9] pipeline compilation failed");
}

SHADER_PARAMS_BEGIN(SpdParams)
    SHADER_PARAM_UAV   (BufferRef, matsOut);
    SHADER_PARAM_SRV   (BufferRef, matsIn);
    SHADER_PARAM_SCALAR(uint32_t,  count);
SHADER_PARAMS_END();

void GpuSpdProjector9::project(Strategy strategy, Clamp clamp,
                               const std::vector<double>& matsIn, int count,
                               std::vector<double>& matsOut) {
    matsOut.assign(size_t(count) * 81, 0.0);
    auto pso = pso_[static_cast<int>(strategy)][static_cast<int>(clamp)];
    if (!pso.valid() || count == 0) return;

    const size_t bytes = size_t(count) * 81 * sizeof(double);
    auto bIn = createDeviceLocalBuffer(device_, bytes, "spd-in");
    auto bOut = createDeviceLocalBuffer(device_, bytes, "spd-out");

    // upload
    {
        auto staging = device_.createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::HostVisible,
            .usage = BufferDesc::TransferSrc});
        std::memcpy(staging->map(), matsIn.data(), bytes);
        staging->unmap();
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> r{{{0, 0, bytes}}};
        cmd->copyBuffer(staging, bIn, r);
        device_.submitAndWait(*cmd, QueueType::Transfer);
    }

    // dispatch
    {
        auto cmd = device_.beginCommands(QueueType::Compute);
        cmd->memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                           BarrierDesc::AccessTransferWrite, BarrierDesc::AccessShaderRead);
        SpdParams p; p.matsOut = bOut; p.matsIn = bIn; p.count = static_cast<uint32_t>(count);
        cmd->dispatch(pso, p, (count + 63) / 64, 1, 1);
        cmd->memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageTransfer,
                           BarrierDesc::AccessShaderWrite, BarrierDesc::AccessTransferRead);
        device_.submitAndWait(*cmd, QueueType::Compute);
    }

    // download
    {
        auto rb = device_.createBuffer({
            .sizeBytes = bytes, .visibility = BufferDesc::Visibility::Readback,
            .usage = BufferDesc::TransferDst});
        auto cmd = device_.beginCommands(QueueType::Transfer);
        std::array<BufferCopy, 1> r{{{0, 0, bytes}}};
        cmd->copyBuffer(bOut, rb, r);
        device_.submitAndWait(*cmd, QueueType::Transfer);
        std::memcpy(matsOut.data(), rb->map(), bytes);
        rb->unmap();
    }
}

} // namespace ksk::fem::gpu
