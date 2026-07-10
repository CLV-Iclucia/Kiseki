//
// vk-image.h
//

#pragma once

#include <RHI/image.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace ksk::rhi::vulkan {

class VulkanDevice;

class VulkanImage final : public Image {
 public:
  VulkanImage(VulkanDevice* device, const ImageDesc& desc);

  // Swapchain-image ctor: wraps a VkImage + VkImageView NOT owned by VMA.
  // The VkImage belongs to the VkSwapchainKHR; we only own the VkImageView.
  // destroy() will skip vmaDestroyImage but still destroy the view.
  VulkanImage(VulkanDevice* device, VkImage image, VkImageView view,
              const ImageDesc& desc);

  ~VulkanImage() override;

  ImageDesc desc() const override { return m_desc; }

  // Backend-internal accessors.
  VkImage vkHandle() const noexcept { return m_image; }
  VkImageView vkView() const noexcept { return m_view; }
  VkImageAspectFlags aspectMask() const noexcept { return m_aspectMask; }

 protected:
  void destroy() noexcept override;

 private:
  VulkanDevice* m_device;
  ImageDesc m_desc;
  VkImage m_image = VK_NULL_HANDLE;
  VkImageView m_view = VK_NULL_HANDLE;
  VmaAllocation m_allocation = VK_NULL_HANDLE;
  VkImageAspectFlags m_aspectMask = 0;
  bool m_ownsImage = true;  // false for swapchain images
};

class VulkanSampler final : public Sampler {
 public:
  VulkanSampler(VulkanDevice* device, const SamplerDesc& desc);
  ~VulkanSampler() override;

  VkSampler vkHandle() const noexcept { return m_sampler; }

 protected:
  void destroy() noexcept override;

 private:
  VulkanDevice* m_device;
  VkSampler m_sampler = VK_NULL_HANDLE;
};

}  // namespace ksk::rhi::vulkan
