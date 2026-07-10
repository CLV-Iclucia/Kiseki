//
// swapchain.h
// Window-surface swapchain abstraction.
// See docs/rhi-r7-swapchain-plan.md.
//
// Design:
//   RHI does NOT depend on any windowing library (GLFW, SDL, etc.).
//   The caller provides a SurfaceCreateFn callback that creates the
//   platform surface (VkSurfaceKHR for Vulkan). This keeps all windowing
//   knowledge in the App/Renderer layer.
//
// Lifetime contract:
//   The user is responsible for ensuring GPU work referencing swapchain
//   images has completed (via waitFence) before calling recreate() or
//   destroying the Swapchain. RHI does NOT do automatic deferred
//   destruction. See docs/rhi-r7-swapchain-plan.md §0 for rationale.
//

#pragma once

#include <Core/properties.h>
#include <RHI/format.h>
#include <RHI/image.h>
#include <RHI/sync.h>

#include <cstdint>
#include <functional>

namespace ksk::rhi {

// ---- Present mode ----------------------------------------------------------
enum class PresentMode : uint32_t {
  Fifo,       // VSync on. Guaranteed not to tear; may block if queue is full.
  Mailbox,    // VSync on + allow frame drops. Low-latency triple-buffered.
  Immediate,  // VSync off. May tear. Lowest latency.
};

// ---- Surface factory callback ----------------------------------------------
//
// The callback receives the backend's native instance handle (for Vulkan this
// is a VkInstance, cast to void*) and must return the created surface handle
// (VkSurfaceKHR cast to void*). Return nullptr on failure.
//
// Example (App layer, which knows about GLFW):
//
//   desc.surfaceCreator = [window](void* instance) -> void* {
//       VkSurfaceKHR surface = VK_NULL_HANDLE;
//       VkResult r = glfwCreateWindowSurface(
//           (VkInstance)instance, (GLFWwindow*)window, nullptr, &surface);
//       return (r == VK_SUCCESS) ? (void*)surface : nullptr;
//   };
//
using SurfaceCreateFn = std::function<void*(void* nativeInstance)>;

// ---- SwapchainDesc ---------------------------------------------------------
struct SwapchainDesc {
  // Required: callback to create the window surface.
  // RHI calls this exactly once during createSwapchain().
  SurfaceCreateFn surfaceCreator;

  // Initial backbuffer dimensions (in pixels, not screen-coordinates).
  uint32_t width = 800;
  uint32_t height = 600;

  // Preferred surface format. The implementation may fall back to the
  // closest supported format on the device.
  Format format = Format::BGRA8_UNorm_sRGB;

  // Desired number of backbuffer images. Actual count may differ (clamped
  // to device min/max). Typical values: 2 (double-buffered) or 3 (triple).
  uint32_t imageCount = 3;

  // Presentation mode.
  PresentMode presentMode = PresentMode::Fifo;
};

// ---- Swapchain abstract class ----------------------------------------------
//
// Owns the VkSwapchainKHR + backbuffer images. The user drives the frame
// loop explicitly:
//
//   1. waitFence(frameFence[i])   — ensure previous use of slot i is done
//   2. resetFence(frameFence[i])
//   3. acquireNextImage(sem)      — get next backbuffer
//   4. record commands
//   5. submit(cmd, wait=sem, signal=renderDone, fence=frameFence[i])
//   6. present(renderDone)
//
// See docs/rhi-r7-swapchain-plan.md §2 for the full pattern.
//
class Swapchain : public ksk::core::NonCopyable {
 public:
  virtual ~Swapchain() = default;

  // Acquire the next backbuffer image. The returned ImageRef is valid until
  // the next call to acquireNextImage() or recreate() — do NOT hold it
  // across frames.
  //
  // signalSemaphore is signaled when the image is ready for rendering.
  // Pass this semaphore to submit()'s wait list.
  //
  // Returns:
  //   Valid ImageRef on success.
  //   Empty ImageRef if the swapchain is out-of-date (caller should
  //   recreate, then retry).
  virtual ImageRef acquireNextImage(Semaphore& signalSemaphore) = 0;

  // Present the most recently acquired image to the screen.
  //
  // waitSemaphore: the semaphore signaled by the submit that wrote to the
  // backbuffer. Present waits on this before flipping.
  //
  // Returns true on success, false if swapchain is out-of-date or suboptimal
  // (caller should recreate before next frame).
  virtual bool present(Semaphore& waitSemaphore) = 0;

  // Recreate the swapchain with new dimensions (e.g. after window resize).
  //
  // PRECONDITION: all GPU work referencing old backbuffer images must be
  // complete. In practice, call this after waitFence() for all in-flight
  // frame slots. Internally calls vkDeviceWaitIdle as a safety net.
  //
  // After recreate(), previously returned ImageRefs are invalid.
  virtual void recreate(uint32_t newWidth, uint32_t newHeight) = 0;

  // ---- Queries -------------------------------------------------------------
  virtual uint32_t imageCount() const = 0;
  virtual Format   imageFormat() const = 0;
  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;

  // The index of the most recently acquired image (valid between
  // acquireNextImage and present).
  virtual uint32_t currentImageIndex() const = 0;

 protected:
  Swapchain() = default;
};

}  // namespace ksk::rhi
