//
// vk-sync.cc
//

#include "vk-sync.h"

#include "vk-device.h"

namespace ksk::rhi::vulkan {

// ---- VulkanFence -----------------------------------------------------------
VulkanFence::VulkanFence(VulkanDevice* device, VkFence handle)
    : m_device(device), m_handle(handle) {}

VulkanFence::~VulkanFence() {
  if (m_handle != VK_NULL_HANDLE) {
    vkDestroyFence(m_device->vkDevice(), m_handle, nullptr);
    m_handle = VK_NULL_HANDLE;
  }
}

// ---- VulkanSemaphore -------------------------------------------------------
VulkanSemaphore::VulkanSemaphore(VulkanDevice* device, VkSemaphore handle)
    : m_device(device), m_handle(handle) {}

VulkanSemaphore::~VulkanSemaphore() {
  if (m_handle != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_device->vkDevice(), m_handle, nullptr);
    m_handle = VK_NULL_HANDLE;
  }
}

}  // namespace ksk::rhi::vulkan
