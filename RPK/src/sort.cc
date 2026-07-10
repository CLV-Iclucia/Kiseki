// ============================================================================
// src/sort.cc — Sort implementation (LSB radix sort, 4 bits/pass)
// ============================================================================

#include <RPK/sort.h>

#ifndef RPK_SHADER_DIR
#define RPK_SHADER_DIR "."
#endif

namespace ksk::rpk {

using namespace ksk::rhi;

// ---- SHADER_PARAMS ----
SHADER_PARAMS_BEGIN(HistogramParams)
    SHADER_PARAM_SRV   (BufferRef, keys);
    SHADER_PARAM_UAV   (BufferRef, histogram);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
    SHADER_PARAM_SCALAR(uint32_t,  bitOffset);
    SHADER_PARAM_SCALAR(uint32_t,  numGroups);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ScatterParams)
    SHADER_PARAM_SRV   (BufferRef, keysIn);
    SHADER_PARAM_SRV   (BufferRef, valuesIn);
    SHADER_PARAM_SRV   (BufferRef, prefixSums);
    SHADER_PARAM_UAV   (BufferRef, keysOut);
    SHADER_PARAM_UAV   (BufferRef, valuesOut);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
    SHADER_PARAM_SCALAR(uint32_t,  bitOffset);
    SHADER_PARAM_SCALAR(uint32_t,  numGroups);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(ScatterKeysOnlyParams)
    SHADER_PARAM_SRV   (BufferRef, keysIn);
    SHADER_PARAM_SRV   (BufferRef, prefixSums);
    SHADER_PARAM_UAV   (BufferRef, keysOut);
    SHADER_PARAM_SCALAR(uint32_t,  numElements);
    SHADER_PARAM_SCALAR(uint32_t,  bitOffset);
    SHADER_PARAM_SCALAR(uint32_t,  numGroups);
SHADER_PARAMS_END();

Sort::Sort(Device& device, ShaderCompiler& compiler,
           const std::filesystem::path& shaderDir)
    : device_(device)
{
    auto dir = shaderDir.empty() ? std::filesystem::path(RPK_SHADER_DIR) : shaderDir;

    psoHistogram_       = compileComputePipeline(device, compiler, dir / "radix-histogram.hlsl");
    psoScatter_         = compileComputePipeline(device, compiler, dir / "radix-scatter.hlsl");
    psoScatterKeysOnly_ = compileComputePipeline(device, compiler, dir / "radix-scatter-keys.hlsl");

    scan_ = std::make_unique<Scan>(device, compiler, dir);
}

void Sort::ensureBuffers(uint32_t count, bool needValues) {
    if (allocatedCount_ >= count) return;
    allocatedCount_ = count;

    uint32_t numGroups = (count + kTileSize - 1) / kTileSize;
    uint32_t histSize  = numGroups * kNumBuckets * sizeof(uint32_t);

    tempKeys_ = device_.createBuffer({
        .sizeBytes  = count * sizeof(uint32_t),
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferSrc,
        .debugName  = "rpk-sort-temp-keys",
    });

    if (needValues) {
        tempValues_ = device_.createBuffer({
            .sizeBytes  = count * sizeof(uint32_t),
            .visibility = BufferDesc::Visibility::DeviceLocal,
            .usage      = BufferDesc::Storage | BufferDesc::TransferSrc,
            .debugName  = "rpk-sort-temp-values",
        });
    }

    histogramBuf_ = device_.createBuffer({
        .sizeBytes  = histSize,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage | BufferDesc::TransferDst,
        .debugName  = "rpk-sort-histogram",
    });

    prefixSumBuf_ = device_.createBuffer({
        .sizeBytes  = histSize,
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage      = BufferDesc::Storage,
        .debugName  = "rpk-sort-prefix",
    });
}

void Sort::pairs(CommandList& cmd,
                 const BufferRef& keys, const BufferRef& values,
                 uint32_t count) {
    if (count <= 1) return;
    ensureBuffers(count, true);

    uint32_t numGroups = (count + kTileSize - 1) / kTileSize;
    uint32_t histElements = numGroups * kNumBuckets;

    BufferRef keySrc = keys;
    BufferRef keyDst = tempKeys_;
    BufferRef valSrc = values;
    BufferRef valDst = tempValues_;

    for (uint32_t pass = 0; pass < kNumPasses; ++pass) {
        uint32_t bitOffset = pass * kRadixBits;

        cmd.fillBuffer(histogramBuf_, 0u);
        cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                          BarrierDesc::AccessTransferWrite,
                          BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

        HistogramParams hp;
        hp.keys        = keySrc;
        hp.histogram   = histogramBuf_;
        hp.numElements = count;
        hp.bitOffset   = bitOffset;
        hp.numGroups   = numGroups;
        cmd.dispatch(psoHistogram_, hp, numGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

        scan_->exclusive(cmd, ScanOp::Sum, ScalarType::Uint32,
                         histogramBuf_, prefixSumBuf_, histElements);

        ScatterParams sp;
        sp.keysIn      = keySrc;
        sp.valuesIn    = valSrc;
        sp.prefixSums  = prefixSumBuf_;
        sp.keysOut     = keyDst;
        sp.valuesOut   = valDst;
        sp.numElements = count;
        sp.bitOffset   = bitOffset;
        sp.numGroups   = numGroups;
        cmd.dispatch(psoScatter_, sp, numGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

        std::swap(keySrc, keyDst);
        std::swap(valSrc, valDst);
    }

    static_assert(kNumPasses % 2 == 0, "Even passes keep result in original buffer");
}

void Sort::keys(CommandList& cmd, const BufferRef& keysBuf, uint32_t count) {
    if (count <= 1) return;
    ensureBuffers(count, false);

    uint32_t numGroups = (count + kTileSize - 1) / kTileSize;
    uint32_t histElements = numGroups * kNumBuckets;

    BufferRef keySrc = keysBuf;
    BufferRef keyDst = tempKeys_;

    for (uint32_t pass = 0; pass < kNumPasses; ++pass) {
        uint32_t bitOffset = pass * kRadixBits;

        cmd.fillBuffer(histogramBuf_, 0u);
        cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                          BarrierDesc::AccessTransferWrite,
                          BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

        HistogramParams hp;
        hp.keys        = keySrc;
        hp.histogram   = histogramBuf_;
        hp.numElements = count;
        hp.bitOffset   = bitOffset;
        hp.numGroups   = numGroups;
        cmd.dispatch(psoHistogram_, hp, numGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

        scan_->exclusive(cmd, ScanOp::Sum, ScalarType::Uint32,
                         histogramBuf_, prefixSumBuf_, histElements);

        ScatterKeysOnlyParams sp;
        sp.keysIn      = keySrc;
        sp.prefixSums  = prefixSumBuf_;
        sp.keysOut     = keyDst;
        sp.numElements = count;
        sp.bitOffset   = bitOffset;
        sp.numGroups   = numGroups;
        cmd.dispatch(psoScatterKeysOnly_, sp, numGroups, 1, 1);
        cmd.memoryBarrier(BarrierDesc::StageComputeShader, BarrierDesc::StageComputeShader);

        std::swap(keySrc, keyDst);
    }

    static_assert(kNumPasses % 2 == 0);
}

}  // namespace ksk::rpk
