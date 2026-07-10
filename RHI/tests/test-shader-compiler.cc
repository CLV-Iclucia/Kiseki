//
// test-shader-compiler.cc
// R3 verification: HLSL → SPIR-V via DXC. Pure CPU; no Vulkan device.
// See docs/rhi-plan.md §8.1 (test-shader-compiler.cc bullet).
//

#include <RHI/shader-compiler.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>

using namespace ksk::rhi;

namespace {

// SPIR-V magic number, little-endian. SPIR-V spec §3, table 1.
constexpr uint32_t kSpirvMagic = 0x07230203u;

// DXIL container header: starts with "DXBC" (legacy bytecode container reused
// by DXIL). dxil/DxilContainer.h: DxilContainerHeader { uint32_t HeaderFourCC; ... }
// FourCC value is 'DXBC' little-endian = 0x43425844.
constexpr uint32_t kDxilFourCC = 0x43425844u;

// Trivial compute kernel — single SSBO + push constants. Picked to exercise
// resource declarations the way fluid kernels will.
//
// NOTE: DXC's [[vk::push_constant]] applies to a struct-typed global variable
// (not to `cbuffer`). Using it on a cbuffer triggers
//   "'push_constant' attribute only applies to global variables of struct type"
// during -spirv codegen on modern DXC.
constexpr const char* kSaxpyHlsl = R"(
RWStructuredBuffer<float> g_y : register(u0);

struct PushParams { float alpha; uint count; };
[[vk::push_constant]] PushParams pc;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.x >= pc.count) return;
  g_y[tid.x] = pc.alpha;
}
)";

// Source that branches on a preprocessor define so we can verify -D actually
// changes codegen.
constexpr const char* kBranchingHlsl = R"(
RWStructuredBuffer<float> g_out : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
#ifdef DOUBLE_IT
  g_out[tid.x] = float(tid.x) * 2.0f;
#else
  g_out[tid.x] = float(tid.x);
#endif
}
)";

// Read a 32-bit little-endian word from a byte stream.
uint32_t readMagic(const std::vector<std::byte>& bytes) {
  uint32_t magic = 0;
  if (bytes.size() >= 4) std::memcpy(&magic, bytes.data(), 4);
  return magic;
}

}  // namespace

TEST(ShaderCompilerTest, CreatorAvailable) {
  auto c = ShaderCompiler::create();
  // If dxcompiler can't load (e.g. CI without the lib), skip rather than fail.
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";
}

TEST(ShaderCompilerTest, CompilesTrivialComputeToSpirv) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions opts{};
  opts.entryPoint = "main";
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;

  auto result = c->compileHlsl(kSaxpyHlsl, opts);
  ASSERT_TRUE(result.has_value()) << "trivial compute kernel failed to compile";
  ASSERT_GE(result->bytecode.size(), 4u);
  EXPECT_EQ(readMagic(result->bytecode), kSpirvMagic)
      << "expected SPIR-V magic at byte 0";
  // SPIR-V word size is 4. Real shaders are dozens of words; 16 bytes is the
  // minimum-sane sanity threshold (header alone).
  EXPECT_GE(result->bytecode.size(), 16u);
}

TEST(ShaderCompilerTest, ReturnsNulloptOnSyntaxError) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions opts{};
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;

  // Garbage source — DXC should reject this.
  auto result = c->compileHlsl("this is not valid HLSL ;;;;;", opts);
  EXPECT_FALSE(result.has_value());
}

TEST(ShaderCompilerTest, MissingEntryPointFails) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions opts{};
  opts.entryPoint = "doesNotExist";
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;

  auto result = c->compileHlsl(kSaxpyHlsl, opts);
  EXPECT_FALSE(result.has_value());
}

TEST(ShaderCompilerTest, DefinesChangeBytecode) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions baseline{};
  baseline.stage = ShaderStage::Compute;
  baseline.targetBackend = Backend::Vulkan;

  auto without = c->compileHlsl(kBranchingHlsl, baseline);

  ShaderCompileOptions withDefine = baseline;
  withDefine.defines.emplace_back("DOUBLE_IT", "1");
  auto with = c->compileHlsl(kBranchingHlsl, withDefine);

  ASSERT_TRUE(without.has_value());
  ASSERT_TRUE(with.has_value());
  EXPECT_NE(without->bytecode, with->bytecode)
      << "Different defines should produce different bytecode";
}

TEST(ShaderCompilerTest, CacheReturnsIdenticalBytecode) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions opts{};
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;

  auto a = c->compileHlsl(kSaxpyHlsl, opts);
  auto b = c->compileHlsl(kSaxpyHlsl, opts);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->bytecode, b->bytecode);
}

TEST(ShaderCompilerTest, CompileFromFile) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  // Write a temp .hlsl file (testing::TempDir exists on all platforms gtest
  // supports).
  const auto dir = std::filesystem::path(::testing::TempDir());
  const auto path = dir / "rhi-r3-trivial.hlsl";
  {
    std::ofstream f(path);
    f << kSaxpyHlsl;
  }

  ShaderCompileOptions opts{};
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;
  auto result = c->compileHlslFile(path, opts);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(readMagic(result->bytecode), kSpirvMagic);

  std::filesystem::remove(path);
}

TEST(ShaderCompilerTest, IncludeDirResolvesNeighborFile) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  const auto dir = std::filesystem::path(::testing::TempDir()) / "rhi-r3-inc";
  std::filesystem::create_directories(dir);
  const auto headerPath = dir / "neighbor.hlsl";
  const auto mainPath = dir / "main.hlsl";

  {
    std::ofstream f(headerPath);
    f << "#define ALPHA 0.5f\n";
  }
  {
    std::ofstream f(mainPath);
    f << "#include \"neighbor.hlsl\"\n"
      << "RWStructuredBuffer<float> g : register(u0);\n"
      << "[numthreads(1,1,1)] void main(uint3 t : SV_DispatchThreadID) {\n"
      << "  g[t.x] = ALPHA;\n"
      << "}\n";
  }

  ShaderCompileOptions opts{};
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Vulkan;

  // compileHlslFile() should auto-add the file's parent dir to includeDirs.
  auto result = c->compileHlslFile(mainPath, opts);
  ASSERT_TRUE(result.has_value()) << "failed to resolve #include via file path";
  EXPECT_EQ(readMagic(result->bytecode), kSpirvMagic);

  std::filesystem::remove_all(dir);
}

// IDxcCompiler3::Disassemble only handles DXIL; for SPIR-V the implementation
// logs an info note and leaves disassembly empty until SPIRV-Tools lands in
// R4. This test verifies (a) the DX12 path actually produces text, and
// (b) the Vulkan path does NOT — i.e. the deliberate gap is observable.
TEST(ShaderCompilerTest, DisassemblyOnDxilOnly) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  // (a) DX12 path: should produce DXIL disassembly.
  {
    ShaderCompileOptions opts{};
    opts.stage = ShaderStage::Compute;
    opts.targetBackend = Backend::Dx12;
    opts.generateDisassembly = true;

    auto result = c->compileHlsl(kSaxpyHlsl, opts);
    if (!result) {
      GTEST_SKIP() << "DXIL emission unavailable in this build";
    }
    EXPECT_FALSE(result->disassembly.empty())
        << "DX12 disassembly should be populated when generateDisassembly=true";
  }

  // (b) Vulkan path: SPIR-V disassembly via SPIRV-Tools (R6+).
  {
    ShaderCompileOptions opts{};
    opts.stage = ShaderStage::Compute;
    opts.targetBackend = Backend::Vulkan;
    opts.generateDisassembly = true;

    auto result = c->compileHlsl(kSaxpyHlsl, opts);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->disassembly.empty())
        << "SPIR-V disassembly should be populated via SPIRV-Tools (R6+)";
  }
}

// DXIL is the fallback / DX12 path. Sign step requires Windows-only dxil.dll;
// we don't actually load the result, just verify DXC will produce *some*
// DXIL container bytes. Skip on platforms where DXIL emission is too brittle
// in the vcpkg port.
TEST(ShaderCompilerTest, DxilEmitProducesContainer) {
  auto c = ShaderCompiler::create();
  if (!c) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions opts{};
  opts.stage = ShaderStage::Compute;
  opts.targetBackend = Backend::Dx12;

  auto result = c->compileHlsl(kSaxpyHlsl, opts);
  if (!result) GTEST_SKIP() << "DXIL emission unavailable in this build";

  ASSERT_GE(result->bytecode.size(), 4u);
  EXPECT_EQ(readMagic(result->bytecode), kDxilFourCC)
      << "expected DXBC/DXIL FourCC at byte 0";
}
