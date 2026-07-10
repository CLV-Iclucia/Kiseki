//
// vk-triangle/main.cc
// R7 demo app: windowed hello-triangle using RHI Swapchain + explicit sync.
//
// This demonstrates the correct frame loop pattern per
// docs/rhi-r7-swapchain-plan.md §2.
//
// Build:
//   cmake -B build -DKISEKI_BUILD_RHI=ON
//   cmake --build build --target vk-triangle
//
// Requires: Vulkan SDK, GPU with Vulkan 1.3, and a display.
//

#include <RHI/rhi.h>

#include <spdlog/spdlog.h>

// GLFW — the ONLY place in the project where glfw + vulkan co-exist for
// surface creation. RHI itself does NOT link or include GLFW.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace ksk::rhi;

// ---- HLSL sources (embedded) -----------------------------------------------
static const char* kVertexShader = R"(
struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

VSOutput main(uint vertexID : SV_VertexID) {
    // Full-screen triangle positions (clip space).
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

// ---- Main ------------------------------------------------------------------
int main() {
  spdlog::set_level(spdlog::level::info);

  // ---- 1. GLFW init --------------------------------------------------------
  if (!glfwInit()) {
    spdlog::error("Failed to initialize GLFW");
    return EXIT_FAILURE;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // No OpenGL context
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  GLFWwindow* window = glfwCreateWindow(800, 600, "SimCraft — VK Triangle (R7)", nullptr, nullptr);
  if (!window) {
    spdlog::error("Failed to create GLFW window");
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // ---- 2. Create RHI Device ------------------------------------------------
  auto device = Device::create({.enableValidation = true});
  if (!device) {
    spdlog::error("No Vulkan device available");
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // ---- 3. Create Swapchain (surface via callback — RHI doesn't know GLFW) --
  auto swapchain = device->createSwapchain({
      .surfaceCreator = [window](void* instance) -> void* {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkResult r = glfwCreateWindowSurface(
            reinterpret_cast<VkInstance>(instance),
            window, nullptr, &surface);
        return (r == VK_SUCCESS) ? reinterpret_cast<void*>(surface) : nullptr;
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

  // ---- 4. Compile shaders --------------------------------------------------
  auto compiler = ShaderCompiler::create();
  if (!compiler) {
    spdlog::error("ShaderCompiler unavailable (dxcompiler not found?)");
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

  auto vs = device->createShader(vsResult->bytecode, ShaderStage::Vertex, "main");
  auto fs = device->createShader(fsResult->bytecode, ShaderStage::Fragment, "main");

  // ---- 5. Create graphics pipeline -----------------------------------------
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

  // ---- 6. Per-frame sync resources -----------------------------------------
  //
  // Semaphore reuse rules (Vulkan spec):
  //   A binary semaphore used in vkQueuePresentKHR's wait list cannot be
  //   re-signaled until the corresponding swapchain image is re-acquired.
  //
  // Solution (validation hint approach "a"):
  //   - renderDone semaphores: one per SWAPCHAIN IMAGE, indexed by the
  //     image index returned by acquireNextImage. When that image is
  //     re-acquired, it proves the prior present completed -- safe to reuse.
  //   - imageReady semaphores + fences: one per FRAME-IN-FLIGHT slot.
  //     The fence wait guarantees the prior submit consumed the imageReady
  //     semaphore, so it's safe to pass to acquire again.
  //
  // See: https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
  //
  constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
  const uint32_t imageCount = swapchain->imageCount();

  // Per frame-in-flight: imageReady semaphore + fence
  std::vector<std::unique_ptr<Semaphore>> imageReadySems(MAX_FRAMES_IN_FLIGHT);
  std::vector<std::unique_ptr<Fence>> frameFences(MAX_FRAMES_IN_FLIGHT);
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    imageReadySems[i] = device->createSemaphore();
    frameFences[i] = device->createFence();
  }

  // Per swapchain image: renderDone semaphore
  std::vector<std::unique_ptr<Semaphore>> renderDoneSems(imageCount);
  for (uint32_t i = 0; i < imageCount; ++i) {
    renderDoneSems[i] = device->createSemaphore();
  }

  // Signal all fences initially so the first frames don't wait forever.
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    auto cmd = device->beginCommands(QueueType::Graphics);
    device->submit(*cmd, {}, {}, frameFences[i].get());
  }
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    device->waitFence(*frameFences[i]);
  }

  uint32_t frameIdx = 0;
  bool needsRecreate = false;

  // Helper: wait ALL in-flight fences before swapchain recreate so that no
  // GPU work references old backbuffer images. recreate() does
  // vkDeviceWaitIdle internally as a safety net, but explicit fence waits
  // avoid the full pipeline stall on drivers that honour it.
  auto waitAllFrames = [&]() {
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      device->waitFence(*frameFences[i]);
    }
  };

  // Helper: (re-)allocate renderDone semaphores to match current image count.
  auto syncRenderDoneSems = [&]() {
    const uint32_t count = swapchain->imageCount();
    if (renderDoneSems.size() != count) {
      renderDoneSems.resize(count);
      for (uint32_t i = 0; i < count; ++i) {
        if (!renderDoneSems[i]) {
          renderDoneSems[i] = device->createSemaphore();
        }
      }
    }
  };

  // ---- 7. Frame loop -------------------------------------------------------
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Handle minimize (framebuffer size = 0).
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    if (fbW == 0 || fbH == 0) {
      glfwWaitEvents();
      continue;
    }

    // (1) Wait for this frame slot's previous GPU work to finish.
    device->waitFence(*frameFences[frameIdx]);
    device->resetFence(*frameFences[frameIdx]);

    // (resize / out-of-date check) -- after waitFence, before acquire.
    if (needsRecreate ||
        static_cast<uint32_t>(fbW) != swapchain->width() ||
        static_cast<uint32_t>(fbH) != swapchain->height()) {
      waitAllFrames();
      swapchain->recreate(static_cast<uint32_t>(fbW),
                          static_cast<uint32_t>(fbH));
      syncRenderDoneSems();
      needsRecreate = false;
    }

    // (2) Acquire next backbuffer.
    ImageRef backbuffer = swapchain->acquireNextImage(*imageReadySems[frameIdx]);
    if (!backbuffer) {
      waitAllFrames();
      swapchain->recreate(static_cast<uint32_t>(fbW),
                          static_cast<uint32_t>(fbH));
      syncRenderDoneSems();
      needsRecreate = false;
      continue;
    }
    uint32_t imgIdx = swapchain->currentImageIndex();

    // ③ Record commands.
    auto cmd = device->beginCommands(QueueType::Graphics);

    // Transition backbuffer: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
    cmd->barrier({
        .srcStage = BarrierDesc::StageTopOfPipe,
        .dstStage = BarrierDesc::StageColorAttachmentOutput,
        .srcAccess = BarrierDesc::AccessNone,
        .dstAccess = BarrierDesc::AccessColorAttachmentWrite,
        .imageBarriers = {{
            .image = backbuffer,
            .oldLayout = BarrierDesc::ImageBarrier::Layout::Undefined,
            .newLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
        }},
    });

    cmd->beginRenderPass({
        .colorAttachments = {{
            .image = backbuffer,
            .loadOp = RenderPassBeginInfo::Attachment::LoadOp::Clear,
            .storeOp = RenderPassBeginInfo::Attachment::StoreOp::Store,
            .clearValue = ClearValue::makeColorF(0.1f, 0.1f, 0.1f, 1.0f),
        }},
        .renderArea = {0, 0, swapchain->width(), swapchain->height()},
    });

    cmd->bindGraphicsPipeline(pipeline);
    cmd->setViewport({0, 0,
                      static_cast<float>(swapchain->width()),
                      static_cast<float>(swapchain->height()),
                      0.f, 1.f});
    cmd->setScissor({0, 0, swapchain->width(), swapchain->height()});
    cmd->draw(3);

    cmd->endRenderPass();

    // Transition backbuffer: COLOR_ATTACHMENT → PRESENT_SRC.
    cmd->barrier({
        .srcStage = BarrierDesc::StageColorAttachmentOutput,
        .dstStage = BarrierDesc::StageBottomOfPipe,
        .srcAccess = BarrierDesc::AccessColorAttachmentWrite,
        .dstAccess = BarrierDesc::AccessNone,
        .imageBarriers = {{
            .image = backbuffer,
            .oldLayout = BarrierDesc::ImageBarrier::Layout::ColorAttachment,
            .newLayout = BarrierDesc::ImageBarrier::Layout::Present,
        }},
    });

    // (4) Submit. renderDone is indexed by image index (not frame index).
    Semaphore* waits[] = {imageReadySems[frameIdx].get()};
    Semaphore* signals[] = {renderDoneSems[imgIdx].get()};
    device->submit(*cmd, waits, signals, frameFences[frameIdx].get());

    // (5) Present. Wait on the same renderDone semaphore.
    bool presentOk = swapchain->present(*renderDoneSems[imgIdx]);
    if (!presentOk) {
      // Suboptimal or out-of-date -- flag for recreate at start of next frame
      // (after the fence wait ensures GPU is idle for this frame slot).
      needsRecreate = true;
    }

    // (6) Advance frame index.
    frameIdx = (frameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  // ---- 8. Cleanup ----------------------------------------------------------
  // Wait all frames to complete before destroying resources.
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    device->waitFence(*frameFences[i]);
  }
  // Ensure present operations are also complete (waitFence only covers
  // submit; present is a subsequent queue operation). submitAndWait on
  // the graphics queue guarantees the queue is fully idle.
  {
    auto cmd = device->beginCommands(QueueType::Graphics);
    device->submitAndWait(*cmd, QueueType::Graphics);
  }

  // Teardown in dependency order:
  // 1. Sync primitives (no longer in flight)
//  frameFences.clear();
//  renderDoneSems.clear();
//  imageReadySems.clear();
  // 2. Swapchain (destroys surface + backbuffer image views)
//  swapchain.reset();
  // 3. Pipeline (holds ShaderRef via GraphicsPipelineDesc in cache)
  //    Explicitly release our ref; the cache still holds one.
//  pipeline = {};
  // 4. Shaders — also clear psoDesc which holds ShaderRefs!
//  psoDesc.vertexShader = {};
//  psoDesc.fragmentShader = {};
//  vs = {};
//  fs = {};
  // 5. Device (dtor: pipeline cache cleared → pipeline destroyed →
  //    shader refs released → shader modules destroyed → then VkDevice)
//  device.reset();

  glfwDestroyWindow(window);
  glfwTerminate();

  spdlog::info("vk-triangle exited cleanly.");
  return EXIT_SUCCESS;
}
