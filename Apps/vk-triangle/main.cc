//
// vk-triangle/main.cc
// Windowed hello-triangle using RHI SwapchainFrameLoop.
//

#include <RHI/rhi.h>

#include <spdlog/spdlog.h>

// The app layer owns window-system integration. RHI receives only a surface
// callback and does not include/link GLFW itself.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>

using namespace ksk::rhi;

static const char* kVertexShader = R"(
struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

VSOutput main(uint vertexID : SV_VertexID) {
    float2 positions[3] = {
        float2( 0.0,  -0.5),
        float2(-0.5,   0.5),
        float2( 0.5,   0.5)
    };
    float3 colors[3] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0)
    };

    VSOutput output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.color = colors[vertexID];
    return output;
}
)";

static const char* kFragmentShader = R"(
struct PSInput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

float4 main(PSInput input) : SV_Target {
    return float4(input.color, 1.0);
}
)";

int main() {
  spdlog::set_level(spdlog::level::info);

  if (!glfwInit()) {
    spdlog::error("Failed to initialize GLFW");
    return EXIT_FAILURE;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  GLFWwindow* window =
      glfwCreateWindow(800, 600, "SimCraft VK Triangle (RHI)", nullptr, nullptr);
  if (!window) {
    spdlog::error("Failed to create GLFW window");
    glfwTerminate();
    return EXIT_FAILURE;
  }

  auto device = Device::create({.enableValidation = true});
  if (!device) {
    spdlog::error("No Vulkan device available");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  auto swapchain = device->createSwapchain({
      .surfaceCreator = [window](void* instance) -> void* {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkResult result = glfwCreateWindowSurface(
            reinterpret_cast<VkInstance>(instance), window, nullptr, &surface);
        return result == VK_SUCCESS ? reinterpret_cast<void*>(surface) : nullptr;
      },
      .width = 800,
      .height = 600,
      .format = Format::BGRA8_UNorm_sRGB,
      .imageCount = 3,
      .presentMode = PresentMode::Fifo,
  });
  if (!swapchain) {
    spdlog::error("Failed to create swapchain");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  auto compiler = ShaderCompiler::create();
  if (!compiler) {
    spdlog::error("ShaderCompiler unavailable");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  auto vsResult = compiler->compileHlsl(kVertexShader, {
      .entryPoint = "main",
      .stage = ShaderStage::Vertex,
      .targetBackend = Backend::Vulkan,
  });
  auto fsResult = compiler->compileHlsl(kFragmentShader, {
      .entryPoint = "main",
      .stage = ShaderStage::Fragment,
      .targetBackend = Backend::Vulkan,
  });
  if (!vsResult || !fsResult) {
    spdlog::error("Shader compilation failed");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  auto vs = device->createVertexShader(vsResult->bytecode, "main");
  auto fs = device->createFragmentShader(fsResult->bytecode, "main");

  GraphicsPipelineDesc psoDesc{};
  psoDesc.vertexShader = vs;
  psoDesc.fragmentShader = fs;
  psoDesc.topology = GraphicsPipelineDesc::PrimitiveTopology::TriangleList;
  psoDesc.depthTest = false;
  psoDesc.depthWrite = false;
  psoDesc.colorFormats.push_back(swapchain->imageFormat());

  auto pipeline = device->createGraphicsPipeline(psoDesc);
  if (!pipeline) {
    spdlog::error("Failed to create graphics pipeline");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  {
    SwapchainFrameLoop frameLoop(*device, *swapchain, 2);

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      int fbW = 0;
      int fbH = 0;
      glfwGetFramebufferSize(window, &fbW, &fbH);
      if (fbW == 0 || fbH == 0) {
        glfwWaitEvents();
        continue;
      }

      SwapchainFrame frame = frameLoop.beginFrame(static_cast<uint32_t>(fbW),
                                                  static_cast<uint32_t>(fbH));
      if (!frame) {
        continue;
      }

      frame.commands->barrier({
        .srcStage = BarrierDesc::StageTopOfPipe,
        .dstStage = BarrierDesc::StageColorAttachmentOutput,
        .srcAccess = BarrierDesc::AccessNone,
        .dstAccess = BarrierDesc::AccessColorAttachmentWrite,
        .imageBarriers = {{
            .image = frame.backbuffer,
            .oldLayout = BarrierDesc::ImageBarrier::Layout::Undefined,
            .newLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
        }},
      });

      frame.commands->beginRenderPass({
        .colorAttachments = {{
            .image = frame.backbuffer,
            .loadOp = RenderPassBeginInfo::Attachment::LoadOp::Clear,
            .storeOp = RenderPassBeginInfo::Attachment::StoreOp::Store,
            .clearValue = ClearValue::makeColorF(0.1f, 0.1f, 0.1f, 1.0f),
        }},
        .renderArea = {0, 0, frame.width, frame.height},
      });

      frame.commands->bindGraphicsPipeline(pipeline);
      frame.commands->setViewport({
        0,
        0,
        static_cast<float>(frame.width),
        static_cast<float>(frame.height),
        0.0f,
        1.0f,
      });
      frame.commands->setScissor({0, 0, frame.width, frame.height});
      frame.commands->draw(3);

      frame.commands->endRenderPass();

      frame.commands->barrier({
        .srcStage = BarrierDesc::StageColorAttachmentOutput,
        .dstStage = BarrierDesc::StageBottomOfPipe,
        .srcAccess = BarrierDesc::AccessColorAttachmentWrite,
        .dstAccess = BarrierDesc::AccessNone,
        .imageBarriers = {{
            .image = frame.backbuffer,
            .oldLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
            .newLayout = BarrierDesc::ImageBarrier::Layout::Present,
        }},
      });

      frameLoop.endFrame(frame);
    }

    frameLoop.waitIdle();
  }

  swapchain.reset();

  glfwDestroyWindow(window);
  glfwTerminate();

  spdlog::info("vk-triangle exited cleanly.");
  return EXIT_SUCCESS;
}
