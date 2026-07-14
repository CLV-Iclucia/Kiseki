#pragma once

#include <RHI/commands.h>
#include <RHI/image.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ksk::rhi {

class Device;
class Fence;
class Semaphore;
class Swapchain;

struct SwapchainFrame {
  ImageRef backbuffer;
  std::unique_ptr<CommandList> commands;
  uint32_t frameIndex = 0;
  uint32_t imageIndex = 0;
  uint32_t width = 0;
  uint32_t height = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return backbuffer && commands;
  }
};

class SwapchainFrameLoop {
public:
  explicit SwapchainFrameLoop(Device& device, Swapchain& swapchain,
                              uint32_t maxFramesInFlight = 2);
  ~SwapchainFrameLoop();

  SwapchainFrameLoop(const SwapchainFrameLoop&) = delete;
  SwapchainFrameLoop& operator=(const SwapchainFrameLoop&) = delete;

  [[nodiscard]] SwapchainFrame beginFrame(uint32_t framebufferWidth,
                                          uint32_t framebufferHeight);
  void endFrame(SwapchainFrame& frame);

  void requestRecreate() noexcept { m_needsRecreate = true; }
  void waitIdle();

private:
  struct FrameSync {
    std::unique_ptr<Semaphore> imageReady;
    std::unique_ptr<Fence> fence;
  };

  void initializeFrameFences();
  void waitAllFrames();
  void syncRenderDoneSemaphores();
  void recreate(uint32_t width, uint32_t height);

  Device& m_device;
  Swapchain& m_swapchain;
  uint32_t m_maxFramesInFlight = 0;
  uint32_t m_frameIndex = 0;
  bool m_needsRecreate = false;

  std::vector<FrameSync> m_frames;
  std::vector<std::unique_ptr<Semaphore>> m_renderDone;
};

} // namespace ksk::rhi
