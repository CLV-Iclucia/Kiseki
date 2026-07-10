//
// vk-swapchain.h
// Vulkan swapchain implementation.
// See docs/rhi-r7-swapchain-plan.md.
//

#pragma once

#include <RHI/swapchain.h>
#include <vulkan/vulkan.h>

#include <vector>

namespace ksk::rhi::vulkan {

class VulkanDevice;

class VulkanSwapchain final : public Swapchain {
 public:
  VulkanSwapchain(VulkanDevice* device, const SwapchainDesc& desc);
  ~VulkanSwapchain() override;

  // ---- Swapchain interface --------------------------------------------------
  ImageRef acquireNextImage(Semaphore& signalSemaphore) override;
  bool present(Semaphore& waitSemaphore) override;
  void recreate(uint32_t newWidth, uint32_t newHeight) override;

  uint32_t imageCount() const override {
    return static_cast<uint32_t>(m_images.size());
  }
  Format   imageFormat() const override { return m_format; }
  uint32_t width() const override { return m_width; }
  uint32_t height() const override { return m_height; }
  uint32_t currentImageIndex() const override { return m_currentImageIdx; }

 private:
  // Build/rebuild the VkSwapchainKHR and backbuffer ImageRefs.
  void build();

  // Destroy backbuffer image views and release ImageRefs.
  void destroyBackbuffers();

  // Choose surface format from device capabilities.
  VkSurfaceFormatKHR chooseSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR>& available, Format preferred);

  // Choose present mode from device capabilities.
  VkPresentModeKHR choosePresentMode(
      const std::vector<VkPresentModeKHR>& available, PresentMode preferred);

  // Reverse VkFormat → Format mapping (only for swapchain-relevant formats).
  static Format fromVkFormat(VkFormat f);

  VulkanDevice* m_device;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

  // Backbuffer images (VkImage owned by swapchain; VulkanImage in non-owning
  // mode wraps them + owns the VkImageView).
  std::vector<ImageRef> m_images;

  // State
  Format m_format = Format::Undefined;
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  uint32_t m_currentImageIdx = 0;
  uint32_t m_desiredImageCount = 3;
  PresentMode m_desiredPresentMode = PresentMode::Fifo;

  // Queue used for present (must support presentation).
  VkQueue m_presentQueue = VK_NULL_HANDLE;
};

}  // namespace ksk::rhi::vulkan
