//
// vk-buffer.h
//

#pragma once

#include <RHI/buffer.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace ksk::rhi::vulkan {

class VulkanDevice;

class VulkanBuffer final : public Buffer {
 public:
  VulkanBuffer(VulkanDevice* device, const BufferDesc& desc);
  ~VulkanBuffer() override;

  size_t sizeBytes() const override { return m_sizeBytes; }
  void* map() override;
  void unmap() override;

  // Backend-internal accessors.
  VkBuffer vkHandle() const noexcept { return m_buffer; }
  VmaAllocation vmaAllocation() const noexcept { return m_allocation; }
  const BufferDesc& desc() const noexcept { return m_desc; }

 protected:
  void destroy() noexcept override;

 private:
  VulkanDevice* m_device;
  BufferDesc m_desc;
  size_t m_sizeBytes = 0;
  VkBuffer m_buffer = VK_NULL_HANDLE;
  VmaAllocation m_allocation = VK_NULL_HANDLE;
  VmaAllocationInfo m_allocationInfo{};
  void* m_persistentMapped = nullptr;  // non-null when MAPPED bit was set
  bool m_userMapped = false;
};

}  // namespace ksk::rhi::vulkan
