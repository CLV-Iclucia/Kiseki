//
// vk-sync.h
// Vulkan implementations of Fence and Semaphore.
// Single ownership; no ref count (per plan §3.7).
//

#pragma once

#include <RHI/sync.h>
#include <vulkan/vulkan.h>

namespace ksk::rhi::vulkan {

class VulkanDevice;

class VulkanFence final : public Fence {
 public:
  VulkanFence(VulkanDevice* device, VkFence handle);
  ~VulkanFence() override;

  VkFence vkHandle() const noexcept { return m_handle; }

 private:
  VulkanDevice* m_device;
  VkFence m_handle;
};

class VulkanSemaphore final : public Semaphore {
 public:
  VulkanSemaphore(VulkanDevice* device, VkSemaphore handle);
  ~VulkanSemaphore() override;

  VkSemaphore vkHandle() const noexcept { return m_handle; }

 private:
  VulkanDevice* m_device;
  VkSemaphore m_handle;
};

}  // namespace ksk::rhi::vulkan
