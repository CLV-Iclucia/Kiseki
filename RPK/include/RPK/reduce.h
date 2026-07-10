//
// reduce.h — GPU parallel reduction (sum, max, min)
//
// Supports multi-level reduction for arbitrary-size inputs.
// Works with float, double, int32_t, uint32_t.
//
// Usage:
//   rpk::Reduce reducer(device, compiler);
//   reducer.run(cmd, rpk::ReduceOp::Sum, inputBuf, outputBuf, numElements);
//   // outputBuf[0] contains the result after GPU execution.
//

#pragma once

#include <RPK/types.h>
#include <RHI/rhi.h>

#include <cstdint>
#include <filesystem>

namespace ksk::rpk {

// Supported reduction operators.
enum class ReduceOp : uint32_t {
    Sum = 0,
    Max = 1,
    Min = 2,
};

// Reduce — multi-level parallel reduction engine.
//
// Internally uses a two-level (or multi-level) tree reduction:
//   Level 0: per-workgroup reduce → partial results buffer
//   Level 1+: repeat until 1 element remains
//
// The caller is responsible for inserting barriers before/after if the
// input/output buffers are used by other dispatches in the same command list.
//
class Reduce {
public:
    static constexpr uint32_t kWorkgroupSize = 256;

    // Construct and compile all pipeline variants.
    // `shaderDir` defaults to RPK_SHADER_DIR (set by CMake).
    Reduce(rhi::Device& device, rhi::ShaderCompiler& compiler,
           const std::filesystem::path& shaderDir = {});

    // Record reduction commands into `cmd`.
    //
    // - `input`:  source buffer (DeviceLocal, Storage)
    // - `output`: destination buffer (DeviceLocal, Storage) — only [0] is written
    // - `count`:  number of elements to reduce
    // - `op`:     reduction operator
    // - `type`:   element scalar type
    //
    // Internally allocates/reuses a temporary buffer for intermediate results.
    // Multiple reductions within the same command list are safe (barriers inserted).
    void run(rhi::CommandList& cmd,
             ReduceOp op,
             ScalarType type,
             const rhi::BufferRef& input,
             const rhi::BufferRef& output,
             uint32_t count);

private:
    rhi::Device& device_;

    // Pipeline variants indexed by [op][type]
    static constexpr int kNumOps   = 3;
    static constexpr int kNumTypes = 4;
    rhi::PipelineRef psoReduce_[kNumOps][kNumTypes];
    rhi::PipelineRef psoReduceFinal_[kNumOps][kNumTypes];

    // Temp buffer for intermediate partial results (resized as needed)
    rhi::BufferRef tempBuf_;
    uint32_t tempBufSize_ = 0;

    void ensureTempBuffer(uint32_t sizeBytes);
};

}  // namespace ksk::rpk
