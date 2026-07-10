// ============================================================================
// src/scan.cc — Scan implementation (Blelloch three-phase prefix sum)
// ============================================================================

#include <RPK/scan.h>

#ifndef RPK_SHADER_DIR
#define RPK_SHADER_DIR "."
#endif

namespace ksk::rpk {

using namespace ksk::rhi;

static PipelineRef compilePso(Device& device, ShaderCompiler& compiler,
                              const std::filesystem::path& path,
                              const std::vector<std::pair<std::string, std::string>>& defines) {
    ShaderCompileOptions opts;
    opts.entryPoint     = "main";
    opts.defines        = defines;
    return compileComputePipeline(device, compiler, path, opts);
}

static const char* scanOpSuffix(ScanOp op) {
    switch (op) {
        case ScanOp::Sum: return "SUM";
        case ScanOp::Max: return "MAX";
        case ScanOp::Min: return "MIN";
    }
    return "SUM";
}

static const char* scanTypeSuffix(ScalarType type) {
    switch (type) {
        case ScalarType::Float32: return "FLOAT32";
        case ScalarType::Float64: return "FLOAT64";
        case ScalarType::Int32:   return "INT32";
        case ScalarType::Uint32:  return "UINT32";
    }
    return "FLOAT32";
}

Scan::Scan(Device& device, ShaderCompiler& compiler,
           const std::filesystem::path& shaderDir)
    : device_(device)
{
    auto dir = shaderDir.empty() ? std::filesystem::path(RPK_SHADER_DIR) : shaderDir;

    for (int op = 0; op < kNumOps; ++op) {
        for (int ty = 0; ty < kNumTypes; ++ty) {
            std::vector<std::pair<std::string, std::string>> defs = {
                {"SCAN_OP", scanOpSuffix(static_cast<ScanOp>(op))},
                {"SCALAR_TYPE", scanTypeSuffix(static_cast<ScalarType>(ty))},
            };

            psoLocalScan_[op][ty] = compilePso(
                device, compiler, dir / "scan-local.hlsl", defs);
            psoPropagate_[op][ty] = compilePso(
                device, compiler, dir / "scan-propagate.hlsl", defs);
        }
    }
}

void Scan::ensureBlockSumBuffer(int level, uint32_t numElements) {
    uint32_t needed = numElements * sizeof(uint32_t);
    if (blockSumSizes_[level] >= needed) return;
    blockSumSizes_[level] = needed;
    blockSumBufs_[level] = device_.createBuffer({
        .sizeBytes  = needed,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName  = "rpk-scan-blocksums-" + std::to_string(level),
    });
}

// ---- SHADER_PARAMS ----
SHADER_PARAMS_BEGIN(ScanLocalParams)
    SHADER_PARAM_SRV   (BufferRef, inputBuf);
    SHADER_PARAM_UAV   (BufferRef, outputBuf);
    SHADER_PARAM_UAV   (BufferRef, blockSums);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
    SHADER_PARAM_SCALAR(uint32_t,  exclusive);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ScanPropagateParams)
    SHADER_PARAM_UAV   (BufferRef, data);
    SHADER_PARAM_SRV   (BufferRef, blockOffsets);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
SHADER_PARAMS_END();

void Scan::scanRecursive(CommandList& cmd, ScanOp op, ScalarType type,
                         const BufferRef& input, const BufferRef& output,
                         uint32_t count, bool isExclusive, int level) {
    if (count == 0) return;

    int opIdx = static_cast<int>(op);
    int tyIdx = static_cast<int>(type);

    uint32_t numGroups = (count + kElementsPerGroup - 1) / kElementsPerGroup;

    // Ensure block sum buffer for this level
    ensureBlockSumBuffer(level, numGroups);

    // Phase 1: local scan per workgroup, output block sums
    ScanLocalParams p1;
    p1.inputBuf    = input;
    p1.outputBuf   = output;
    p1.blockSums   = blockSumBufs_[level];
    p1.numElements = count;
    p1.exclusive   = isExclusive ? 1u : 0u;
    cmd.dispatch(psoLocalScan_[opIdx][tyIdx], p1, numGroups, 1, 1);
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    if (numGroups > 1) {
        // Phase 2: recursively scan the block sums (always exclusive)
        if (numGroups <= kElementsPerGroup) {
            ScanLocalParams p2;
            p2.inputBuf    = blockSumBufs_[level];
            p2.outputBuf   = blockSumBufs_[level];
            p2.blockSums   = blockSumBufs_[level];
            p2.numElements = numGroups;
            p2.exclusive   = 1u;
            cmd.dispatch(psoLocalScan_[opIdx][tyIdx], p2, 1, 1, 1);
            cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
        } else {
            int nextLevel = level + 1;
            ensureBlockSumBuffer(nextLevel, (numGroups + kElementsPerGroup - 1) / kElementsPerGroup);
            scanRecursive(cmd, op, type,
                          blockSumBufs_[level], blockSumBufs_[level],
                          numGroups, true, nextLevel);
        }

        // Phase 3: propagate block offsets to local elements
        ScanPropagateParams p3;
        p3.data         = output;
        p3.blockOffsets = blockSumBufs_[level];
        p3.numElements  = count;
        cmd.dispatch(psoPropagate_[opIdx][tyIdx], p3, numGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
    }
}

void Scan::exclusive(CommandList& cmd, ScanOp op, ScalarType type,
                     const BufferRef& input, const BufferRef& output,
                     uint32_t count) {
    scanRecursive(cmd, op, type, input, output, count, true, 0);
}

void Scan::inclusive(CommandList& cmd, ScanOp op, ScalarType type,
                     const BufferRef& input, const BufferRef& output,
                     uint32_t count) {
    scanRecursive(cmd, op, type, input, output, count, false, 0);
}

}  // namespace ksk::rpk
