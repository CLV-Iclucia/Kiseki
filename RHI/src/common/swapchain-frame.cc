#include <RHI/swapchain-frame.h>

#include <RHI/device.h>
#include <RHI/swapchain.h>
#include <RHI/sync.h>

#include <algorithm>

namespace ksk::rhi {

SwapchainFrameLoop::SwapchainFrameLoop(Device& device, Swapchain& swapchain,
                                       uint32_t maxFramesInFlight)
    : m_device(device),
      m_swapchain(swapchain),
      m_maxFramesInFlight(std::max(maxFramesInFlight, 1u)),
      m_frames(m_maxFramesInFlight) {
  for (auto& frame : m_frames) {
    frame.imageReady = m_device.createSemaphore();
    frame.fence = m_device.createFence();
  }

  syncRenderDoneSemaphores();
  initializeFrameFences();
}

SwapchainFrameLoop::~SwapchainFrameLoop() {
  waitIdle();
}

SwapchainFrame SwapchainFrameLoop::beginFrame(uint32_t framebufferWidth,
                                              uint32_t framebufferHeight) {
  if (framebufferWidth == 0 || framebufferHeight == 0) {
    return {};
  }

  auto& sync = m_frames[m_frameIndex];
  m_device.waitFence(*sync.fence);
  m_device.resetFence(*sync.fence);

  if (m_needsRecreate ||
      framebufferWidth != m_swapchain.width() ||
      framebufferHeight != m_swapchain.height()) {
    recreate(framebufferWidth, framebufferHeight);
  }

  ImageRef backbuffer = m_swapchain.acquireNextImage(*sync.imageReady);
  if (!backbuffer) {
    recreate(framebufferWidth, framebufferHeight);
    return {};
  }

  uint32_t imageIndex = m_swapchain.currentImageIndex();
  if (imageIndex >= m_renderDone.size()) {
    syncRenderDoneSemaphores();
  }

  return {
      .backbuffer = backbuffer,
      .commands = m_device.beginCommands(QueueType::Graphics),
      .frameIndex = m_frameIndex,
      .imageIndex = imageIndex,
      .width = m_swapchain.width(),
      .height = m_swapchain.height(),
  };
}

void SwapchainFrameLoop::endFrame(SwapchainFrame& frame) {
  if (!frame) return;

  auto& sync = m_frames[frame.frameIndex];
  Semaphore* waits[] = {sync.imageReady.get()};
  Semaphore* signals[] = {m_renderDone[frame.imageIndex].get()};

  m_device.submit(*frame.commands, waits, signals, sync.fence.get());

  bool presentOk = m_swapchain.present(*m_renderDone[frame.imageIndex]);
  if (!presentOk) {
    m_needsRecreate = true;
  }

  frame.commands.reset();
  frame.backbuffer = {};
  m_frameIndex = (m_frameIndex + 1) % m_maxFramesInFlight;
}

void SwapchainFrameLoop::waitIdle() {
  waitAllFrames();

  // Fences cover rendering submissions. Submit a no-op and wait for the queue
  // so any queued present waits are also drained before teardown/recreate.
  auto cmd = m_device.beginCommands(QueueType::Graphics);
  m_device.submitAndWait(*cmd, QueueType::Graphics);
}

void SwapchainFrameLoop::initializeFrameFences() {
  for (auto& frame : m_frames) {
    auto cmd = m_device.beginCommands(QueueType::Graphics);
    m_device.submit(*cmd, {}, {}, frame.fence.get());
  }

  waitAllFrames();
}

void SwapchainFrameLoop::waitAllFrames() {
  for (auto& frame : m_frames) {
    if (frame.fence) {
      m_device.waitFence(*frame.fence);
    }
  }
}

void SwapchainFrameLoop::syncRenderDoneSemaphores() {
  const uint32_t imageCount = m_swapchain.imageCount();
  m_renderDone.resize(imageCount);
  for (auto& sem : m_renderDone) {
    if (!sem) {
      sem = m_device.createSemaphore();
    }
  }
}

void SwapchainFrameLoop::recreate(uint32_t width, uint32_t height) {
  waitAllFrames();
  m_swapchain.recreate(width, height);
  syncRenderDoneSemaphores();
  m_needsRecreate = false;
}

} // namespace ksk::rhi
