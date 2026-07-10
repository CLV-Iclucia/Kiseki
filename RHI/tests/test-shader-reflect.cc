//
// test-shader-reflect.cc
// R4 verification: SPIR-V → ReflectionInfo via SPIRV-Reflect.
// See docs/rhi-plan.md §8.1 (test-shader-reflect.cc bullet) and §3.4.1.
//
// Exercises a HLSL kernel covering all five descriptor types we model:
// StorageBuffer (UAV), SampledImage (SRV), Sampler, UniformBuffer (cbuffer),
// plus push constants. The kernel is fed through DXC at runtime so the test
// stays self-contained — if dxcompiler is absent the test GTEST_SKIPs the
// same way test-shader-compiler does.
//

#include <RHI/reflection.h>
#include <RHI/shader-compiler.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace ksk::rhi;

namespace {

// HLSL kernel using explicit [[vk::binding(N, S)]] annotations so the
// resulting SPIR-V (set, binding) numbers are deterministic across DXC
// versions — DXC's *default* register→binding mapping otherwise has historic
// drift between versions.
constexpr const char* kFullCoverageHlsl = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> g_y;
[[vk::binding(1, 0)]] Texture2D                 g_tex;
[[vk::binding(2, 0)]] SamplerState              g_samp;
[[vk::binding(3, 0)]] cbuffer Frame { float4x4 g_view; };

struct PushParams { float alpha; uint count; };
[[vk::push_constant]] PushParams pc;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.x >= pc.count) return;
  float2 uv = float2(0.0, 0.0);
  // Reference every binding so DXC can't dead-strip.
  float v = pc.alpha
          + g_tex.SampleLevel(g_samp, uv, 0).x
          + g_view[0][0];
  g_y[tid.x] = v;
}
)";

// Compile helper. Returns std::nullopt if dxcompiler is unavailable, in which
// case the caller should GTEST_SKIP.
std::optional<CompiledShader> compile(std::string_view source) {
  auto compiler = ShaderCompiler::create();
  if (!compiler) return std::nullopt;
  ShaderCompileOptions opts;
  opts.entryPoint = "main";
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;
  return compiler->compileHlsl(source, opts);
}

// Find a binding by (set, binding). Returns nullptr if not present.
const DescriptorBindingInfo* findBinding(const ReflectionInfo& ri,
                                         uint32_t set, uint32_t binding) {
  for (const auto& b : ri.bindings) {
    if (b.set == set && b.binding == binding) return &b;
  }
  return nullptr;
}

}  // namespace

TEST(ShaderReflect, EmptyBlobReturnsEmpty) {
  // No DXC needed — purely tests reflectSpirv's bail-out path.
  ReflectionInfo ri = reflectSpirv({}, ShaderStage::Compute, "main");
  EXPECT_EQ(ri.stage, ShaderStage::Compute);
  EXPECT_EQ(ri.entryPoint, "main");
  EXPECT_TRUE(ri.bindings.empty());
  EXPECT_TRUE(ri.pushConstants.empty());
}

TEST(ShaderReflect, FullCoverageKernel) {
  auto compiled = compile(kFullCoverageHlsl);
  if (!compiled) GTEST_SKIP() << "dxcompiler unavailable; skipping reflection test";
  ASSERT_FALSE(compiled->bytecode.empty());

  ReflectionInfo ri = reflectSpirv(compiled->bytecode, ShaderStage::Compute, "main");

  EXPECT_EQ(ri.stage, ShaderStage::Compute);
  EXPECT_EQ(ri.entryPoint, "main");

  // All four descriptor bindings should be present.
  ASSERT_EQ(ri.bindings.size(), 4u) << "expected 4 bindings (UAV, SRV, sampler, cbuffer)";

  using Type = DescriptorBindingInfo::Type;

  const auto* yBuf = findBinding(ri, 0, 0);
  ASSERT_NE(yBuf, nullptr);
  EXPECT_EQ(yBuf->type, Type::StorageBuffer);
  EXPECT_EQ(yBuf->count, 1u);

  const auto* tex = findBinding(ri, 0, 1);
  ASSERT_NE(tex, nullptr);
  EXPECT_EQ(tex->type, Type::SampledImage);
  EXPECT_EQ(tex->count, 1u);

  const auto* samp = findBinding(ri, 0, 2);
  ASSERT_NE(samp, nullptr);
  EXPECT_EQ(samp->type, Type::Sampler);
  EXPECT_EQ(samp->count, 1u);

  const auto* cb = findBinding(ri, 0, 3);
  ASSERT_NE(cb, nullptr);
  EXPECT_EQ(cb->type, Type::UniformBuffer);
  EXPECT_EQ(cb->count, 1u);

  // Push constants: { float alpha; uint count; } — 8 bytes total at offset 0.
  ASSERT_EQ(ri.pushConstants.size(), 1u);
  EXPECT_EQ(ri.pushConstants[0].offset, 0u);
  EXPECT_EQ(ri.pushConstants[0].size, 8u);
}

TEST(ShaderReflect, NamesPreservedWhenAvailable) {
  auto compiled = compile(kFullCoverageHlsl);
  if (!compiled) GTEST_SKIP() << "dxcompiler unavailable; skipping reflection test";
  ASSERT_FALSE(compiled->bytecode.empty());

  ReflectionInfo ri = reflectSpirv(compiled->bytecode, ShaderStage::Compute, "main");
  ASSERT_EQ(ri.bindings.size(), 4u);

  // SPIRV-Reflect surfaces the HLSL variable name verbatim. The check is
  // "every binding has SOME non-empty name" rather than asserting exact
  // strings — DXC has historically prefixed cbuffers with "type." in some
  // versions, and that's not material to R4's correctness.
  for (const auto& b : ri.bindings) {
    EXPECT_FALSE(b.name.empty())
        << "binding (set=" << b.set << ", binding=" << b.binding << ") has no name";
  }

  // The compute kernel binding ordering should be stable enough that g_y is
  // findable by name.
  bool sawGy = std::any_of(ri.bindings.begin(), ri.bindings.end(),
                           [](const DescriptorBindingInfo& b) {
                             return b.name.find("g_y") != std::string::npos;
                           });
  EXPECT_TRUE(sawGy) << "expected to find a binding whose name contains 'g_y'";
}

TEST(ShaderReflect, MissingEntryPointReturnsEmpty) {
  auto compiled = compile(kFullCoverageHlsl);
  if (!compiled) GTEST_SKIP() << "dxcompiler unavailable; skipping reflection test";
  ASSERT_FALSE(compiled->bytecode.empty());

  // Wrong entry point — reflectSpirv should log and return an empty result
  // (not crash, not throw).
  ReflectionInfo ri = reflectSpirv(compiled->bytecode, ShaderStage::Compute,
                                   "this_entry_does_not_exist");
  EXPECT_EQ(ri.entryPoint, "this_entry_does_not_exist");
  EXPECT_TRUE(ri.bindings.empty());
  EXPECT_TRUE(ri.pushConstants.empty());
}

TEST(ShaderReflect, MinimalKernelOnlyOneBinding) {
  // Sanity: a kernel with a single UAV (no sampler, no cbuffer, no PC) yields
  // exactly one binding and zero push constants.
  constexpr const char* kMinimal = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> g_y;
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  g_y[tid.x] = 1.0;
}
)";
  auto compiled = compile(kMinimal);
  if (!compiled) GTEST_SKIP() << "dxcompiler unavailable; skipping reflection test";
  ASSERT_FALSE(compiled->bytecode.empty());

  ReflectionInfo ri = reflectSpirv(compiled->bytecode, ShaderStage::Compute, "main");
  ASSERT_EQ(ri.bindings.size(), 1u);
  EXPECT_EQ(ri.bindings[0].set, 0u);
  EXPECT_EQ(ri.bindings[0].binding, 0u);
  EXPECT_EQ(ri.bindings[0].type, DescriptorBindingInfo::Type::StorageBuffer);
  EXPECT_TRUE(ri.pushConstants.empty());
}
