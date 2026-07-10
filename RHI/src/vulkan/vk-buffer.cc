//
// vk-buffer.cc
//

#include "vk-buffer.h"

#include "vk-device.h"
#include "vk-internals.h"

namespace ksk::rhi::vulkan {

VulkanBuffer::VulkanBuffer(VulkanDevice* device, const BufferDesc& desc)
    : m_device(device), m_desc(desc), m_sizeBytes(desc.sizeBytes) {
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = desc.sizeBytes;
  bi.usage = toVkBufferUsage(desc.usage);
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo ai{};
  ai.usage = toVmaMemoryUsage(desc.visibility);
  ai.flags = toVmaAllocationFlags(desc.visibility);

  VK_CHECK(vmaCreateBuffer(m_device->vmaAllocator(), &bi, &ai, &m_buffer,
                           &m_allocation, &m_allocationInfo));

  if (ai.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
    m_persistentMapped = m_allocationInfo.pMappedData;
  }

  if (m_device->validationEnabled() && !desc.debugName.empty()) {
    VkDebugUtilsObjectNameInfoEXT ni{};
    ni.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    ni.objectType = VK_OBJECT_TYPE_BUFFER;
    ni.objectHandle = (uint64_t)m_buffer;
    ni.pObjectName = desc.debugName.c_str();
    auto fn = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
        m_device->vkDevice(), "vkSetDebugUtilsObjectNameEXT");
    if (fn) fn(m_device->vkDevice(), &ni);
  }
}

VulkanBuffer::~VulkanBuffer() {
  if (m_userMapped && !m_persistentMapped) {
    vmaUnmapMemory(m_device->vmaAllocator(), m_allocation);
  }
  if (m_buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_device->vmaAllocator(), m_buffer, m_allocation);
    m_buffer = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
  }
}

void* VulkanBuffer::map() {
  if (m_persistentMapped) return m_persistentMapped;
  void* p = nullptr;
  VK_CHECK(vmaMapMemory(m_device->vmaAllocator(), m_allocation, &p));
  m_userMapped = true;
  return p;
}

void VulkanBuffer::unmap() {
  if (m_persistentMapped) return;
  if (m_userMapped) {
    vmaUnmapMemory(m_device->vmaAllocator(), m_allocation);
    m_userMapped = false;
  }
}

void VulkanBuffer::destroy() noexcept {
  // Tier 0 path: synchronously wait for any in-flight GPU work that might
  // still reference this buffer, then delete. R0–R2 has no swapchain so
  // frameLoopActive() is always false; Tier 1 lives in R8.
  if (!m_device->frameLoopActive()) {
    vkDeviceWaitIdle(m_device->vkDevice());
    delete this;
    return;
  }
  // Future R8: m_device->deferDestroy(this, m_lastUseFrame);
  delete this;
}

}  // namespace ksk::rhi::vulkan
