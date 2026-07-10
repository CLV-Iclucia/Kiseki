//
// test-compute-trivial.cc
// R5 verification: end-to-end compute via SHADER_PARAMS + three-in-one
// dispatch. The "P0 GPU compute path opened" sign-off per plan §13.8.
//
// Flow:
//   1. Compile saxpy HLSL → SPIR-V via the R3 ShaderCompiler.
//   2. Device::createShader runs SPIRV-Reflect (R4) to build ReflectionInfo.
//   3. Device::createComputePipeline derives layout from reflection (R4).
//   4. Upload Y values to a GPU storage buffer via staging.
//   5. cmd->dispatch(pso, params, gx) — three-in-one path:
//        bindComputePipeline → resolve(first time) → apply → rawDispatch.
//   6. Copy GPU buffer to readback, verify Y[i] = alpha (for in-bounds i).
//
// GTEST_SKIP if no Vulkan device or no dxcompiler.
//

#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace ksk::rhi;

// ---- HLSL kernel ----------------------------------------------------------
//
// The simplest meaningful compute: write `alpha` into the first `count`
// elements of a UAV buffer. Single set, single binding, two-scalar push
// constant — exercises every R4/R5 code path without dragging in textures.
namespace {

constexpr const char* kSaxpyHlsl = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> g_y;

struct PushParams { uint count; float alpha; };
[[vk::push_constant]] PushParams pc;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.x >= pc.count) return;
  g_y[tid.x] = pc.alpha;
}
)";

constexpr uint32_t kElems = 256;

inline uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

// ---- SHADER_PARAMS at namespace scope (static-init of _<field>_reg) ------
SHADER_PARAMS_BEGIN(SaxpyComputeParams)
  SHADER_PARAM_UAV   (BufferRef, g_y);
  SHADER_PARAM_SCALAR(uint32_t,  count);
  SHADER_PARAM_SCALAR(float,     alpha);
SHADER_PARAMS_END();

TEST(ComputeTrivial, SaxpyEndToEndViaShaderParams) {
  // ---- Compile ---------------------------------------------------------
  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "dxcompiler unavailable; skipping";

  ShaderCompileOptions copts;
  copts.entryPoint = "main";
  copts.stage = ShaderStage::Compute;
  copts.targetBackend = Backend::Vulkan;
  auto compiled = compiler->compileHlsl(kSaxpyHlsl, copts);
  ASSERT_TRUE(compiled.has_value());
  ASSERT_FALSE(compiled->bytecode.empty());

  // ---- Device ----------------------------------------------------------
  auto device =
      Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  // ---- Shader + pipeline ----------------------------------------------
  auto shader = device->createShader(compiled->bytecode, ShaderStage::Compute,
                                     "main");
  ASSERT_TRUE(shader.valid());

  // Sanity-check reflection saw what we declared.
  const auto& ri = shader->reflection();
  ASSERT_EQ(ri.bindings.size(), 1u);
  EXPECT_EQ(ri.bindings[0].set, 0u);
  EXPECT_EQ(ri.bindings[0].binding, 0u);
  EXPECT_EQ(ri.bindings[0].type,
            DescriptorBindingInfo::Type::StorageBuffer);
  ASSERT_EQ(ri.pushConstants.size(), 1u);
  EXPECT_EQ(ri.pushConstants[0].size, 8u);

  ComputePipelineDesc pd;
  pd.shader = shader;
  auto pso = device->createComputePipeline(pd);
  ASSERT_TRUE(pso.valid());

  // ---- Buffers ---------------------------------------------------------
  const size_t bytes = kElems * sizeof(float);

  auto gpu = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::DeviceLocal,
      .usage = BufferDesc::Storage | BufferDesc::TransferSrc |
               BufferDesc::TransferDst,
      .debugName = "saxpy-gpu",
  });
  auto readback = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
      .debugName = "saxpy-readback",
  });
  ASSERT_TRUE(gpu.valid());
  ASSERT_TRUE(readback.valid());

  // ---- SHADER_PARAMS instance ----------------------------------------
  SaxpyComputeParams params;
  params.g_y = gpu;
  params.count = kElems;
  params.alpha = 2.5f;

  // ---- Record + submit ------------------------------------------------
  auto cmd = device->beginCommands(QueueType::Compute);

  // Zero out gpu before dispatch — otherwise validation may complain about
  // reads of uninitialised memory in the unused tail beyond `count`. (We
  // wrote `count == kElems` so the kernel actually fills everything; this
  // is just defensive against future smaller `count` runs.)
  cmd->fillBuffer(gpu, 0u);
  {
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageTransfer;
    b.dstStage = BarrierDesc::StageComputeShader;
    b.srcAccess = BarrierDesc::AccessTransferWrite;
    b.dstAccess = BarrierDesc::AccessShaderWrite;
    cmd->barrier(b);
  }

  // The marquee call.
  cmd->dispatch(pso, params, divRoundUp(kElems, 64), 1, 1);

  // Compute → transfer barrier before readback.
  {
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageComputeShader;
    b.dstStage = BarrierDesc::StageTransfer;
    b.srcAccess = BarrierDesc::AccessShaderWrite;
    b.dstAccess = BarrierDesc::AccessTransferRead;
    cmd->barrier(b);
  }

  std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
  cmd->copyBuffer(gpu, readback, region);

  device->submitAndWait(*cmd, QueueType::Compute);

  // ---- Verify ---------------------------------------------------------
  auto out = readback->mapTyped<float>();
  ASSERT_EQ(out.size(), kElems);
  for (uint32_t i = 0; i < kElems; ++i) {
    EXPECT_FLOAT_EQ(out[i], 2.5f) << "mismatch at i=" << i;
  }
  readback->unmap();
}

// ---- Pipeline reuse: same params → second dispatch hits the resolved
//      cache (no re-resolve), produces the right result.
TEST(ComputeTrivial, SaxpyReuseSameParams) {
  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "dxcompiler unavailable; skipping";

  auto compiled = compiler->compileHlsl(
      kSaxpyHlsl,
      {.entryPoint = "main",
       .stage = ShaderStage::Compute,
       .targetBackend = Backend::Vulkan});
  ASSERT_TRUE(compiled.has_value());

  auto device =
      Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  auto shader = device->createShader(compiled->bytecode, ShaderStage::Compute,
                                     "main");
  auto pso = device->createComputePipeline({.shader = shader});

  const size_t bytes = kElems * sizeof(float);
  auto gpu = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::DeviceLocal,
      .usage = BufferDesc::Storage | BufferDesc::TransferSrc,
  });
  auto readback = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
  });

  SaxpyComputeParams params;
  params.g_y = gpu;
  params.count = kElems;

  // First dispatch with alpha=1.0
  params.alpha = 1.0f;
  {
    auto cmd = device->beginCommands(QueueType::Compute);
    cmd->dispatch(pso, params, divRoundUp(kElems, 64), 1, 1);
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageComputeShader;
    b.dstStage = BarrierDesc::StageTransfer;
    b.srcAccess = BarrierDesc::AccessShaderWrite;
    b.dstAccess = BarrierDesc::AccessTransferRead;
    cmd->barrier(b);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(gpu, readback, region);
    device->submitAndWait(*cmd, QueueType::Compute);
  }
  {
    auto out = readback->mapTyped<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    readback->unmap();
  }
  EXPECT_EQ(params._resolvedFor, shader.get())
      << "_resolvedFor should be set after first dispatch";

  // Second dispatch with alpha=4.5 — params reused, _resolve must NOT run
  // again (sentinel matches).
  params.alpha = 4.5f;
  {
    auto cmd = device->beginCommands(QueueType::Compute);
    cmd->dispatch(pso, params, divRoundUp(kElems, 64), 1, 1);
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageComputeShader;
    b.dstStage = BarrierDesc::StageTransfer;
    b.srcAccess = BarrierDesc::AccessShaderWrite;
    b.dstAccess = BarrierDesc::AccessTransferRead;
    cmd->barrier(b);
    std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
    cmd->copyBuffer(gpu, readback, region);
    device->submitAndWait(*cmd, QueueType::Compute);
  }
  {
    auto out = readback->mapTyped<float>();
    for (uint32_t i = 0; i < kElems; ++i) {
      EXPECT_FLOAT_EQ(out[i], 4.5f);
    }
    readback->unmap();
  }
  EXPECT_EQ(params._resolvedFor, shader.get())
      << "_resolvedFor should remain stable when shader didn't change";
}
