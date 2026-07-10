#include <RHI/reflection.h>
#include <RHI/shader-compiler.h>
#include <RHI/structured-buffer-layout.h>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <span>
#include <string_view>

using namespace ksk::rhi;

namespace {

constexpr std::string_view kAccessShader = R"(
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] StructuredBuffer<float> inputVectors;
[[vk::binding(1, 0)]] StructuredBuffer<float> inputMatrices;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputVectors;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputMatrices;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float3 vectorValue = load_float3(inputVectors, tid.x);
    float3x3 matrixValue =
        load_float3x3_col_major(inputMatrices, tid.x);
    store_float3(outputVectors, tid.x, mul(matrixValue, vectorValue));
    store_float3x3_col_major(
        outputMatrices, tid.x, matrixValue + float3x3(
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0));
}
)";

}  // namespace

TEST(StructuredBufferLayout, PackedGlmTypesExposeExactBytes) {
  const std::array<glm::vec3, 2> vectors{{
      {1.0f, -2.0f, 3.5f},
      {4.25f, 5.0f, -6.75f},
  }};

  const auto bytes =
      asStructuredBytes(std::span<const glm::vec3>(vectors));
  ASSERT_EQ(bytes.size(), vectors.size() * 3 * sizeof(float));

  std::array<float, 6> scalars{};
  std::memcpy(scalars.data(), bytes.data(), bytes.size());
  EXPECT_FLOAT_EQ(scalars[0], 1.0f);
  EXPECT_FLOAT_EQ(scalars[1], -2.0f);
  EXPECT_FLOAT_EQ(scalars[2], 3.5f);
  EXPECT_FLOAT_EQ(scalars[3], 4.25f);
  EXPECT_FLOAT_EQ(scalars[4], 5.0f);
  EXPECT_FLOAT_EQ(scalars[5], -6.75f);
}

TEST(StructuredBufferLayout, GlmMat3IsColumnMajorAndTightlyPacked) {
  glm::mat3 matrix(0.0f);
  matrix[0] = glm::vec3(1.0f, 2.0f, 3.0f);
  matrix[1] = glm::vec3(4.0f, 5.0f, 6.0f);
  matrix[2] = glm::vec3(7.0f, 8.0f, 9.0f);

  const std::array<glm::mat3, 1> matrices{{matrix}};
  const auto bytes =
      asStructuredBytes(std::span<const glm::mat3>(matrices));
  ASSERT_EQ(bytes.size(), 9 * sizeof(float));

  std::array<float, 9> scalars{};
  std::memcpy(scalars.data(), bytes.data(), bytes.size());
  for (size_t i = 0; i < scalars.size(); ++i) {
    EXPECT_FLOAT_EQ(scalars[i], static_cast<float>(i + 1));
  }
}

TEST(StructuredBufferLayout, SharedHlslAccessHeaderCompiles) {
  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

  ShaderCompileOptions options;
  options.entryPoint = "main";
  options.stage = ShaderStage::Compute;
  options.targetBackend = Backend::Vulkan;

  auto compiled = compiler->compileHlsl(kAccessShader, options);
  ASSERT_TRUE(compiled.has_value());
  ASSERT_FALSE(compiled->bytecode.empty());

  const ReflectionInfo reflection =
      reflectSpirv(compiled->bytecode, ShaderStage::Compute, "main");
  EXPECT_EQ(reflection.bindings.size(), 4u);
}
