// ============================================================================
// src/reduce.cc — Reduce implementation
// ============================================================================

#include <RPK/reduce.h>

#ifndef RPK_SHADER_DIR
#define RPK_SHADER_DIR "."
#endif

namespace sim::rpk {

using namespace sim::rhi;

// Helper: compile compute pipeline with custom defines (compileComputePipeline
// in shader-utils.h doesn't accept ShaderCompileOptions).
static PipelineRef compilePso(Device& device, ShaderCompiler& compiler,
                              const std::filesystem::path& path,
                              const std::vector<std::pair<std::string, std::string>>& defines) {
    ShaderCompileOptions opts;
    opts.entryPoint     = "main";
    opts.defines        = defines;
    return compileComputePipeline(device, compiler, path, opts);
}

static const char* opSuffix(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum: return "SUM";
        case ReduceOp::Max: return "MAX";
        case ReduceOp::Min: return "MIN";
    }
    return "SUM";
}

static const char* typeSuffix(ScalarType type) {
    switch (type) {
        case ScalarType::Float32: return "FLOAT32";
        case ScalarType::Float64: return "FLOAT64";
        case ScalarType::Int32:   return "INT32";
        case ScalarType::Uint32:  return "UINT32";
    }
    return "FLOAT32";
}

static uint32_t scalarSize(ScalarType type) {
    switch (type) {
        case ScalarType::Float32: return 4;
        case ScalarType::Float64: return 8;
        case ScalarType::Int32:   return 4;
        case ScalarType::Uint32:  return 4;
    }
    return 4;
}

Reduce::Reduce(Device& device, ShaderCompiler& compiler,
               const std::filesystem::path& shaderDir)
    : device_(device)
{
    auto dir = shaderDir.empty() ? std::filesystem::path(RPK_SHADER_DIR) : shaderDir;

    for (int op = 0; op < kNumOps; ++op) {
        for (int ty = 0; ty < kNumTypes; ++ty) {
            std::vector<std::pair<std::string, std::string>> defs = {
                {"REDUCE_OP", opSuffix(static_cast<ReduceOp>(op))},
                {"SCALAR_TYPE", typeSuffix(static_cast<ScalarType>(ty))},
            };

            psoReduce_[op][ty] = compilePso(
                device, compiler, dir / "reduce.hlsl", defs);
            psoReduceFinal_[op][ty] = compilePso(
                device, compiler, dir / "reduce-final.hlsl", defs);
        }
    }
}

void Reduce::ensureTempBuffer(uint32_t sizeBytes) {
    if (tempBufSize_ >= sizeBytes) return;
    tempBufSize_ = sizeBytes;
    tempBuf_ = device_.createBuffer({
        .sizeBytes  = sizeBytes,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName  = "rpk-reduce-temp",
    });
}

// ---- SHADER_PARAMS for reduce dispatches ----
SHADER_PARAMS_BEGIN(ReduceParams)
    SHADER_PARAM_SRV   (BufferRef, inputBuf);
    SHADER_PARAM_UAV   (BufferRef, outputBuf);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
SHADER_PARAMS_END();

void Reduce::run(CommandList& cmd, ReduceOp op, ScalarType type,
                 const BufferRef& input, const BufferRef& output,
                 uint32_t count) {
    if (count == 0) return;

    int opIdx = static_cast<int>(op);
    int tyIdx = static_cast<int>(type);
    uint32_t elemSize = scalarSize(type);

    uint32_t numGroups = (count + kWorkgroupSize - 1) / kWorkgroupSize;

    if (numGroups == 1) {
        // Single workgroup — write directly to output
        ReduceParams p;
        p.inputBuf    = input;
        p.outputBuf   = output;
        p.numElements = count;
        cmd.dispatch(psoReduceFinal_[opIdx][tyIdx], p, 1, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
        return;
    }

    // Multi-level: reduce to partial results, then final reduce
    uint32_t tempSize = numGroups * elemSize;
    ensureTempBuffer(tempSize);

    // Pass 1: per-workgroup reduction
    ReduceParams p1;
    p1.inputBuf    = input;
    p1.outputBuf   = tempBuf_;
    p1.numElements = count;
    cmd.dispatch(psoReduce_[opIdx][tyIdx], p1, numGroups, 1, 1);
    cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

    // Pass 2: final reduction of partial results
    if (numGroups > kWorkgroupSize) {
        // Recursive: reduce tempBuf_ → output
        run(cmd, op, type, tempBuf_, output, numGroups);
    } else {
        ReduceParams p2;
        p2.inputBuf    = tempBuf_;
        p2.outputBuf   = output;
        p2.numElements = numGroups;
        cmd.dispatch(psoReduceFinal_[opIdx][tyIdx], p2, 1, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);
    }
}

}  // namespace sim::rpk
