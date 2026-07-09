//
// sort.h — GPU parallel radix sort (key-value pairs)
//
// 32-bit key radix sort, 4 bits per pass (8 passes total).
// Each pass: histogram → exclusive scan → scatter.
//
// Supports key-only and key-value pair sorting.
// Value type is always uint32_t (can be used as index into other arrays).
//
// Usage:
//   rpk::Sort sorter(device, compiler);
//   sorter.pairs(cmd, keysBuf, valuesBuf, numElements);
//   // After execution: keys sorted ascending, values permuted accordingly.
//
//   // Key-only:
//   sorter.keys(cmd, keysBuf, numElements);
//

#pragma once

#include <RPK/scan.h>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace sim::rpk {

// Sort — LSB radix sort engine (32-bit keys, 4 bits/pass, 8 passes).
//
// Algorithm per pass:
//   1. Histogram: count occurrences of each digit (16 buckets) per workgroup tile
//   2. Scan: exclusive prefix sum over the global histogram
//   3. Scatter: place elements at their sorted positions
//
// Double-buffered: alternates between user buffer and internal temp buffer.
// Final result is always in the user-provided buffer (copied back if needed).
//
class Sort {
public:
    static constexpr uint32_t kWorkgroupSize  = 256;
    static constexpr uint32_t kRadixBits      = 4;
    static constexpr uint32_t kNumBuckets     = 1 << kRadixBits;  // 16
    static constexpr uint32_t kNumPasses      = 32 / kRadixBits;  // 8
    static constexpr uint32_t kTileSize       = kWorkgroupSize;    // elements per WG

    Sort(rhi::Device& device, rhi::ShaderCompiler& compiler,
         const std::filesystem::path& shaderDir = {});

    // Sort key-value pairs in ascending order by key.
    // `keys`:   buffer of uint32_t[count]
    // `values`: buffer of uint32_t[count] (permuted alongside keys)
    void pairs(rhi::CommandList& cmd,
               const rhi::BufferRef& keys,
               const rhi::BufferRef& values,
               uint32_t count);

    // Sort keys only (no associated values).
    void keys(rhi::CommandList& cmd,
              const rhi::BufferRef& keys,
              uint32_t count);

    // Access the internal Scan instance (useful for sharing).
    Scan& scan() { return *scan_; }

private:
    rhi::Device& device_;

    rhi::PipelineRef psoHistogram_;
    rhi::PipelineRef psoScatter_;
    rhi::PipelineRef psoScatterKeysOnly_;

    // Internal scan engine for histogram prefix sums
    std::unique_ptr<Scan> scan_;

    // Double buffers for ping-pong
    rhi::BufferRef tempKeys_;
    rhi::BufferRef tempValues_;
    rhi::BufferRef histogramBuf_;   // [numWorkgroups * kNumBuckets]
    rhi::BufferRef prefixSumBuf_;   // [numWorkgroups * kNumBuckets]

    uint32_t allocatedCount_ = 0;

    void ensureBuffers(uint32_t count, bool needValues);
};

}  // namespace sim::rpk
