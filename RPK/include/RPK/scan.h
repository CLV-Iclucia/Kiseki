//
// scan.h — GPU parallel exclusive/inclusive prefix sum
//
// Implements the Blelloch three-phase scan:
//   Phase 1: per-workgroup local scan + write block sums
//   Phase 2: scan the block sums (recursive for large inputs)
//   Phase 3: propagate block offsets back to local elements
//
// Supports: uint32_t, int32_t, float (sum only for float).
//
// Usage:
//   rpk::Scan scanner(device, compiler);
//   scanner.exclusive(cmd, rpk::ScanOp::Sum, rpk::ScalarType::Uint32,
//                     inputBuf, outputBuf, numElements);
//

#pragma once

#include <RPK/types.h>
#include <RHI/rhi.h>

#include <cstdint>
#include <filesystem>

namespace ksk::rpk {

enum class ScanOp : uint32_t {
    Sum = 0,
    Max = 1,
    Min = 2,
};

// Scan — Blelloch-style parallel prefix sum engine.
//
// Exclusive scan: output[i] = op(input[0], ..., input[i-1])  (output[0] = identity)
// Inclusive scan: output[i] = op(input[0], ..., input[i])
//
class Scan {
public:
    static constexpr uint32_t kWorkgroupSize = 256;
    static constexpr uint32_t kItemsPerThread = 16;
    // Each workgroup processes kWorkgroupSize * kItemsPerThread elements
    static constexpr uint32_t kElementsPerGroup = kWorkgroupSize * kItemsPerThread;

    Scan(rhi::Device& device, rhi::ShaderCompiler& compiler,
         const std::filesystem::path& shaderDir = {});

    // Exclusive prefix sum: output[0] = identity, output[i] = op(input[0..i-1])
    void exclusive(rhi::CommandList& cmd,
                   ScanOp op,
                   ScalarType type,
                   const rhi::BufferRef& input,
                   const rhi::BufferRef& output,
                   uint32_t count);

    // Inclusive prefix sum: output[i] = op(input[0..i])
    void inclusive(rhi::CommandList& cmd,
                  ScanOp op,
                  ScalarType type,
                  const rhi::BufferRef& input,
                  const rhi::BufferRef& output,
                  uint32_t count);

private:
    rhi::Device& device_;

    // Pipelines indexed by [op][type]
    static constexpr int kNumOps   = 3;
    static constexpr int kNumTypes = 4;

    rhi::PipelineRef psoLocalScan_[kNumOps][kNumTypes];       // Phase 1: local scan
    rhi::PipelineRef psoPropagate_[kNumOps][kNumTypes];       // Phase 3: propagate

    // Temp buffers for block sums (multi-level)
    static constexpr int kMaxLevels = 4;  // supports up to 512^4 ≈ 68B elements
    rhi::BufferRef blockSumBufs_[kMaxLevels];
    uint32_t blockSumSizes_[kMaxLevels] = {};

    void ensureBlockSumBuffer(int level, uint32_t numElements);

    void scanRecursive(rhi::CommandList& cmd, ScanOp op, ScalarType type,
                       const rhi::BufferRef& input, const rhi::BufferRef& output,
                       uint32_t count, bool isExclusive, int level);
};

}  // namespace ksk::rpk
