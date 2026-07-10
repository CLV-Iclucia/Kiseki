#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <type_traits>

using namespace ksk::rhi;

namespace {

std::filesystem::path typedFillShaderPath() {
  return std::filesystem::path(::testing::TempDir()) /
         "rhi-typed-fill.hlsl";
}

class TypedFillCS final : public ComputeShader<TypedFillCS> {
 public:
  struct Permutation {
    uint32_t value = 1;
  };

  DECLARE_COMPUTE_SHADER(TypedFillCS);

  SHADER_PARAMS_BEGIN(Params)
    SHADER_PARAM_UAV   (BufferRef, output);
    SHADER_PARAM_SCALAR(uint32_t,  count);
  SHADER_PARAMS_END();

  static void modifyCompileOptions(ShaderCompileOptions& options,
                                   const Permutation& permutation) {
    options.defines.emplace_back("OUTPUT_VALUE",
                                 std::to_string(permutation.value));
  }
};

IMPLEMENT_COMPUTE_SHADER(TypedFillCS, typedFillShaderPath(), "main");

class WrongParamsCS final : public ComputeShader<WrongParamsCS> {
 public:
  DECLARE_COMPUTE_SHADER(WrongParamsCS);

  SHADER_PARAMS_BEGIN(Params)
    SHADER_PARAM_UAV(BufferRef, differentName);
  SHADER_PARAMS_END();
};

IMPLEMENT_COMPUTE_SHADER(WrongParamsCS, typedFillShaderPath(), "main");

static_assert(!std::is_copy_constructible_v<TypedFillCS::Params>);
static_assert(!std::is_move_constructible_v<TypedFillCS::Params>);
static_assert(std::is_copy_constructible_v<TypedFillCS>);
static_assert(std::is_move_constructible_v<TypedFillCS>);

constexpr const char* kTypedFillHlsl = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;

struct PushParams { uint count; };
[[vk::push_constant]] PushParams pc;

#ifndef OUTPUT_VALUE
#define OUTPUT_VALUE 1
#endif

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.x >= pc.count) return;
  output[tid.x] = OUTPUT_VALUE;
}
)";

void writeTypedFillShader() {
  std::ofstream file(typedFillShaderPath(), std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << kTypedFillHlsl;
}

}  // namespace

TEST(TypedComputeShader, CachesProgramAndPipelineByFinalCompileOptions) {
  writeTypedFillShader();

  auto device =
      Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  TypedFillCS::Permutation seven{.value = 7};
  TypedFillCS first(*device, seven);
  if (!first) GTEST_SKIP() << "dxcompiler unavailable";

  TypedFillCS second(*device, seven);
  ASSERT_TRUE(second.valid());
  EXPECT_EQ(&first.reflection(), &second.reflection())
      << "same compile options should reuse the cached pipeline";

  TypedFillCS::Permutation nine{.value = 9};
  TypedFillCS different(*device, nine);
  ASSERT_TRUE(different.valid());
  EXPECT_NE(&first.reflection(), &different.reflection())
      << "different permutations must not alias the program cache";

  EXPECT_THROW(
      {
        WrongParamsCS wrong(*device);
        (void)wrong;
      },
      std::runtime_error)
      << "typed shader construction should validate its nested Params schema";

  std::filesystem::remove(typedFillShaderPath());
}

TEST(TypedComputeShader, DispatchesWithNestedParams) {
  writeTypedFillShader();

  auto device =
      Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  constexpr uint32_t kCount = 256;
  TypedFillCS::Permutation permutation{.value = 37};
  TypedFillCS shader(*device, permutation);
  if (!shader) GTEST_SKIP() << "dxcompiler unavailable";

  auto output = device->createBuffer({
      .sizeBytes = kCount * sizeof(uint32_t),
      .visibility = BufferDesc::Visibility::DeviceLocal,
      .usage = BufferDesc::Storage | BufferDesc::TransferSrc,
      .debugName = "typed-fill-output",
  });
  auto readback = device->createBuffer({
      .sizeBytes = kCount * sizeof(uint32_t),
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
      .debugName = "typed-fill-readback",
  });
  ASSERT_TRUE(output.valid());
  ASSERT_TRUE(readback.valid());

  TypedFillCS::Params params;
  params.output = output;
  params.count = kCount;

  auto cmd = device->beginCommands(QueueType::Compute);
  cmd->dispatch(shader, params, kCount / 64);
  cmd->memoryBarrier(BarrierDesc::StageComputeShader,
                     BarrierDesc::StageTransfer,
                     BarrierDesc::AccessShaderWrite,
                     BarrierDesc::AccessTransferRead);

  std::array<BufferCopy, 1> copy{{{0, 0, kCount * sizeof(uint32_t)}}};
  cmd->copyBuffer(output, readback, copy);
  device->submitAndWait(*cmd, QueueType::Compute);

  auto values = readback->mapTyped<uint32_t>();
  ASSERT_EQ(values.size(), kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(values[i], 37u) << "mismatch at index " << i;
  }
  readback->unmap();

  std::filesystem::remove(typedFillShaderPath());
}
