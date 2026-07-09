#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace sim::rhi;

namespace {

constexpr const char* kComputeShader = R"(
RWStructuredBuffer<uint> output : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  output[tid.x] = tid.x;
}
)";

} // namespace

TEST(ShaderUtilsTest, PipelineBackendAlwaysMatchesDevice) {
  auto device = Device::create({
      .backend = Backend::Vulkan,
      .enableValidation = true,
  });
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "dxcompiler unavailable";

  const auto path =
      std::filesystem::path(::testing::TempDir()) /
      "rhi-compute-pipeline-backend.hlsl";
  {
    std::ofstream file(path);
    file << kComputeShader;
  }

  ShaderCompileOptions options;
  options.stage = ShaderStage::Fragment;
  options.targetBackend = Backend::Dx12;

  auto pipeline = compileComputePipeline(*device, *compiler, path, options);
  std::filesystem::remove(path);

  EXPECT_TRUE(pipeline.valid())
      << "compileComputePipeline must override stage and backend from Device";
}
