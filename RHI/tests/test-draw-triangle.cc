//
// test-draw-triangle.cc
// R6: End-to-end graphics pipeline test — renders a full-screen red triangle
// to an offscreen RGBA8 image, reads back via staging buffer, and verifies
// that the center pixel is red.
//

#include <RHI/rhi.h>

#include <gtest/gtest.h>

#include <cstring>

using namespace ksk::rhi;

SHADER_PARAMS_BEGIN(GraphicsPushParams)
  SHADER_PARAM_VEC(float4, offset);
  SHADER_PARAM_VEC(float4, color);
SHADER_PARAMS_END();

TEST(DrawTriangle, RedTriangleOffscreen) {
  auto device = Device::create({.backend = Backend::Vulkan,
                                .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device";

  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "No dxcompiler";

  // 1. Compile shaders
  static const char* vsSource = R"(
    struct VSInput  { float3 pos : POSITION; };
    struct VSOutput { float4 pos : SV_Position; };
    VSOutput main(VSInput input) {
      VSOutput o;
      o.pos = float4(input.pos, 1.0);
      return o;
    }
  )";
  static const char* fsSource = R"(
    float4 main() : SV_Target { return float4(1, 0, 0, 1); }
  )";

  auto vsBlob = compiler->compileHlsl(vsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Vertex,
      .targetBackend = Backend::Vulkan});
  auto fsBlob = compiler->compileHlsl(fsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Fragment,
      .targetBackend = Backend::Vulkan});
  ASSERT_TRUE(vsBlob.has_value()) << "VS compile failed";
  ASSERT_TRUE(fsBlob.has_value()) << "FS compile failed";

  auto vs = device->createShader(vsBlob->bytecode, ShaderStage::Vertex, "main");
  auto fs = device->createShader(fsBlob->bytecode, ShaderStage::Fragment, "main");
  ASSERT_TRUE(vs);
  ASSERT_TRUE(fs);

  // 2. Create graphics pipeline
  GraphicsPipelineDesc desc;
  desc.vertexShader = vs;
  desc.fragmentShader = fs;
  desc.vertexBindings = {{
      .binding = 0,
      .stride = sizeof(float) * 3,
      .attributes = {{.location = 0, .format = Format::RGB32_Float, .offset = 0}},
  }};
  desc.topology = GraphicsPipelineDesc::PrimitiveTopology::TriangleList;
  desc.depthTest = false;
  desc.depthWrite = false;
  desc.colorFormats = {Format::RGBA8_UNorm};

  auto pso = device->createGraphicsPipeline(desc);
  ASSERT_TRUE(pso);

  // 3. Create resources
  const uint32_t W = 64, H = 64;
  auto colorImg = device->createImage({
      .dim = ImageDesc::Dim::D2,
      .width = W,
      .height = H,
      .depth = 1,
      .format = Format::RGBA8_UNorm,
      .usage = ImageDesc::ColorAttachment | ImageDesc::TransferSrc,
  });
  ASSERT_TRUE(colorImg);

  // Full-screen oversized triangle covering entire NDC.
  // Vulkan clip space has Y pointing down, so CCW winding in screen space
  // means we order vertices to be counter-clockwise when Y is flipped.
  float verts[] = {
      -1.f,  3.f, 0.f,
       3.f, -1.f, 0.f,
      -1.f, -1.f, 0.f,
  };
  auto vbuf = device->createBuffer({
      .sizeBytes = sizeof(verts),
      .visibility = BufferDesc::Visibility::HostVisible,
      .usage = BufferDesc::Vertex,
  });
  ASSERT_TRUE(vbuf);
  std::memcpy(vbuf->map(), verts, sizeof(verts));

  auto readback = device->createBuffer({
      .sizeBytes = W * H * 4,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
  });
  ASSERT_TRUE(readback);

  // 4. Record commands
  auto cmd = device->beginCommands(QueueType::Graphics);

  // Transition: Undefined → ColorAttachment
  cmd->barrier({
      .srcStage = BarrierDesc::StageTopOfPipe,
      .dstStage = BarrierDesc::StageColorAttachmentOutput,
      .srcAccess = BarrierDesc::AccessNone,
      .dstAccess = BarrierDesc::AccessColorAttachmentWrite,
      .imageBarriers = {{
          .image = colorImg,
          .oldLayout = BarrierDesc::ImageBarrier::Layout::Undefined,
          .newLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
      }},
  });

  cmd->beginRenderPass({
      .colorAttachments = {{
          .image = colorImg,
          .loadOp = RenderPassBeginInfo::Attachment::LoadOp::Clear,
          .storeOp = RenderPassBeginInfo::Attachment::StoreOp::Store,
          .clearValue = ClearValue::makeColorF(0, 0, 0, 1),
      }},
      .renderArea = {0, 0, W, H},
  });

  cmd->bindGraphicsPipeline(pso);
  cmd->setViewport({0, 0, static_cast<float>(W), static_cast<float>(H), 0, 1});
  cmd->setScissor({0, 0, W, H});
  cmd->bindVertexBuffer(0, vbuf);
  cmd->draw(3);

  cmd->endRenderPass();

  // Transition: ColorAttachment → TransferSrc
  cmd->barrier({
      .srcStage = BarrierDesc::StageColorAttachmentOutput,
      .dstStage = BarrierDesc::StageTransfer,
      .srcAccess = BarrierDesc::AccessColorAttachmentWrite,
      .dstAccess = BarrierDesc::AccessTransferRead,
      .imageBarriers = {{
          .image = colorImg,
          .oldLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
          .newLayout = BarrierDesc::ImageBarrier::Layout::TransferSrc,
      }},
  });

  BufferImageCopy copyRegion{};
  copyRegion.imageExtentW = W;
  copyRegion.imageExtentH = H;
  copyRegion.imageExtentD = 1;
  cmd->copyImageToBuffer(colorImg, readback, {&copyRegion, 1});

  device->submitAndWait(*cmd, QueueType::Graphics);

  // 5. Verify center pixel is red
  auto* pixels = static_cast<const uint8_t*>(readback->map());
  size_t center = (H / 2 * W + W / 2) * 4;
  EXPECT_EQ(pixels[center + 0], 255);  // R
  EXPECT_EQ(pixels[center + 1], 0);    // G
  EXPECT_EQ(pixels[center + 2], 0);    // B
  EXPECT_EQ(pixels[center + 3], 255);  // A
}

TEST(DrawTriangle, GraphicsShaderParamsPushConstants) {
  auto device = Device::create({.backend = Backend::Vulkan,
                                .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device";

  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "No dxcompiler";

  static const char* vsSource = R"(
    struct PushParams {
      float4 offset;
      float4 color;
    };
    [[vk::push_constant]] PushParams pc;

    struct VSInput  { float3 pos : POSITION; };
    struct VSOutput { float4 pos : SV_Position; };

    VSOutput main(VSInput input) {
      VSOutput o;
      o.pos = float4(input.pos.xy + pc.offset.xy, input.pos.z, 1.0);
      return o;
    }
  )";

  static const char* fsSource = R"(
    struct PushParams {
      float4 offset;
      float4 color;
    };
    [[vk::push_constant]] PushParams pc;

    float4 main() : SV_Target {
      return pc.color;
    }
  )";

  auto vsBlob = compiler->compileHlsl(vsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Vertex,
      .targetBackend = Backend::Vulkan});
  auto fsBlob = compiler->compileHlsl(fsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Fragment,
      .targetBackend = Backend::Vulkan});
  ASSERT_TRUE(vsBlob.has_value()) << "VS compile failed";
  ASSERT_TRUE(fsBlob.has_value()) << "FS compile failed";

  auto vs = device->createShader(vsBlob->bytecode, ShaderStage::Vertex, "main");
  auto fs = device->createShader(fsBlob->bytecode, ShaderStage::Fragment, "main");
  ASSERT_TRUE(vs);
  ASSERT_TRUE(fs);

  GraphicsPipelineDesc desc;
  desc.vertexShader = vs;
  desc.fragmentShader = fs;
  desc.vertexBindings = {{
      .binding = 0,
      .stride = sizeof(float) * 3,
      .attributes = {{.location = 0, .format = Format::RGB32_Float, .offset = 0}},
  }};
  desc.topology = GraphicsPipelineDesc::PrimitiveTopology::TriangleList;
  desc.depthTest = false;
  desc.depthWrite = false;
  desc.colorFormats = {Format::RGBA8_UNorm};

  auto pso = device->createGraphicsPipeline(desc);
  ASSERT_TRUE(pso);

  const uint32_t W = 64;
  const uint32_t H = 64;
  auto colorImg = device->createImage({
      .dim = ImageDesc::Dim::D2,
      .width = W,
      .height = H,
      .depth = 1,
      .format = Format::RGBA8_UNorm,
      .usage = ImageDesc::ColorAttachment | ImageDesc::TransferSrc,
  });
  ASSERT_TRUE(colorImg);

  float verts[] = {
      -1.f,  3.f, 0.f,
       3.f, -1.f, 0.f,
      -1.f, -1.f, 0.f,
  };
  auto vbuf = device->createBuffer({
      .sizeBytes = sizeof(verts),
      .visibility = BufferDesc::Visibility::HostVisible,
      .usage = BufferDesc::Vertex,
  });
  ASSERT_TRUE(vbuf);
  std::memcpy(vbuf->map(), verts, sizeof(verts));

  auto readback = device->createBuffer({
      .sizeBytes = W * H * 4,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
  });
  ASSERT_TRUE(readback);

  GraphicsPushParams params;
  params.offset = float4(0.0f, 0.0f, 0.0f, 0.0f);
  params.color = float4(0.0f, 1.0f, 0.0f, 1.0f);

  auto cmd = device->beginCommands(QueueType::Graphics);
  cmd->barrier({
      .srcStage = BarrierDesc::StageTopOfPipe,
      .dstStage = BarrierDesc::StageColorAttachmentOutput,
      .srcAccess = BarrierDesc::AccessNone,
      .dstAccess = BarrierDesc::AccessColorAttachmentWrite,
      .imageBarriers = {{
          .image = colorImg,
          .oldLayout = BarrierDesc::ImageBarrier::Layout::Undefined,
          .newLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
      }},
  });

  cmd->beginRenderPass({
      .colorAttachments = {{
          .image = colorImg,
          .loadOp = RenderPassBeginInfo::Attachment::LoadOp::Clear,
          .storeOp = RenderPassBeginInfo::Attachment::StoreOp::Store,
          .clearValue = ClearValue::makeColorF(0, 0, 0, 1),
      }},
      .renderArea = {0, 0, W, H},
  });

  cmd->setViewport({0, 0, static_cast<float>(W), static_cast<float>(H), 0, 1});
  cmd->setScissor({0, 0, W, H});
  cmd->bindVertexBuffer(0, vbuf);
  cmd->draw(pso, params, 3);

  cmd->endRenderPass();

  cmd->barrier({
      .srcStage = BarrierDesc::StageColorAttachmentOutput,
      .dstStage = BarrierDesc::StageTransfer,
      .srcAccess = BarrierDesc::AccessColorAttachmentWrite,
      .dstAccess = BarrierDesc::AccessTransferRead,
      .imageBarriers = {{
          .image = colorImg,
          .oldLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
          .newLayout = BarrierDesc::ImageBarrier::Layout::TransferSrc,
      }},
  });

  BufferImageCopy copyRegion{};
  copyRegion.imageExtentW = W;
  copyRegion.imageExtentH = H;
  copyRegion.imageExtentD = 1;
  cmd->copyImageToBuffer(colorImg, readback, {&copyRegion, 1});

  device->submitAndWait(*cmd, QueueType::Graphics);

  auto* pixels = static_cast<const uint8_t*>(readback->map());
  size_t center = (H / 2 * W + W / 2) * 4;
  EXPECT_EQ(pixels[center + 0], 0);
  EXPECT_EQ(pixels[center + 1], 255);
  EXPECT_EQ(pixels[center + 2], 0);
  EXPECT_EQ(pixels[center + 3], 255);
}

TEST(DrawTriangle, PipelineCacheSameDesc) {
  auto device = Device::create({.backend = Backend::Vulkan});
  if (!device) GTEST_SKIP() << "No Vulkan device";

  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "No dxcompiler";

  static const char* vsSource = R"(
    struct VSInput  { float3 pos : POSITION; };
    struct VSOutput { float4 pos : SV_Position; };
    VSOutput main(VSInput input) {
      VSOutput o;
      o.pos = float4(input.pos, 1.0);
      return o;
    }
  )";
  static const char* fsSource = R"(
    float4 main() : SV_Target { return float4(1, 0, 0, 1); }
  )";

  auto vsBlob = compiler->compileHlsl(vsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Vertex,
      .targetBackend = Backend::Vulkan});
  auto fsBlob = compiler->compileHlsl(fsSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Fragment,
      .targetBackend = Backend::Vulkan});
  ASSERT_TRUE(vsBlob && fsBlob);

  auto vs = device->createShader(vsBlob->bytecode, ShaderStage::Vertex, "main");
  auto fs = device->createShader(fsBlob->bytecode, ShaderStage::Fragment, "main");

  GraphicsPipelineDesc desc;
  desc.vertexShader = vs;
  desc.fragmentShader = fs;
  desc.vertexBindings = {{
      .binding = 0,
      .stride = sizeof(float) * 3,
      .attributes = {{.location = 0, .format = Format::RGB32_Float, .offset = 0}},
  }};
  desc.depthTest = false;
  desc.depthWrite = false;
  desc.colorFormats = {Format::RGBA8_UNorm};

  auto pso1 = device->createGraphicsPipeline(desc);
  auto pso2 = device->createGraphicsPipeline(desc);

  // Pipeline cache should return the same pipeline for identical desc
  EXPECT_EQ(pso1.get(), pso2.get());
}

TEST(DrawTriangle, ComputePipelineCache) {
  auto device = Device::create({.backend = Backend::Vulkan});
  if (!device) GTEST_SKIP() << "No Vulkan device";

  auto compiler = ShaderCompiler::create();
  if (!compiler) GTEST_SKIP() << "No dxcompiler";

  static const char* csSource = R"(
    RWStructuredBuffer<float> output : register(u0);
    [numthreads(64,1,1)]
    void main(uint3 id : SV_DispatchThreadID) { output[id.x] = 42.0; }
  )";

  auto csBlob = compiler->compileHlsl(csSource, {
      .entryPoint = "main",
      .stage = ShaderStage::Compute,
      .targetBackend = Backend::Vulkan});
  ASSERT_TRUE(csBlob);

  auto cs = device->createShader(csBlob->bytecode, ShaderStage::Compute, "main");

  ComputePipelineDesc desc;
  desc.shader = cs;

  auto pso1 = device->createComputePipeline(desc);
  auto pso2 = device->createComputePipeline(desc);

  // Pipeline cache should return the same pipeline for identical desc
  EXPECT_EQ(pso1.get(), pso2.get());
}
