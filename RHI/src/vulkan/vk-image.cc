//
// vk-image.cc
//

#include "vk-image.h"

#include "vk-device.h"
#include "vk-internals.h"

namespace ksk::rhi::vulkan {

// ---- VulkanImage -----------------------------------------------------------
VulkanImage::VulkanImage(VulkanDevice* device, const ImageDesc& desc)
    : m_device(device), m_desc(desc) {
  m_aspectMask = aspectMaskFor(desc.format);

  VkImageCreateInfo ii{};
  ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ii.imageType = toVkImageType(desc.dim);
  ii.format = toVkFormat(desc.format);
  ii.extent.width = desc.width;
  ii.extent.height = desc.height;
  ii.extent.depth = (desc.dim == ImageDesc::Dim::D3) ? desc.depth : 1u;
  ii.mipLevels = desc.mipLevels;
  ii.arrayLayers = (desc.dim == ImageDesc::Dim::D3) ? 1u : desc.arrayLayers;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = toVkImageUsage(desc.usage);
  ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo ai{};
  ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VK_CHECK(vmaCreateImage(m_device->vmaAllocator(), &ii, &ai, &m_image,
                          &m_allocation, nullptr));

  // Default view spanning all mips & array layers.
  VkImageViewCreateInfo vi{};
  vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vi.image = m_image;
  vi.viewType = toVkImageViewType(desc.dim, ii.arrayLayers);
  vi.format = ii.format;
  vi.subresourceRange.aspectMask = m_aspectMask;
  vi.subresourceRange.baseMipLevel = 0;
  vi.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
  vi.subresourceRange.baseArrayLayer = 0;
  vi.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
  VK_CHECK(vkCreateImageView(m_device->vkDevice(), &vi, nullptr, &m_view));

  if (m_device->validationEnabled() && !desc.debugName.empty()) {
    VkDebugUtilsObjectNameInfoEXT ni{};
    ni.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    ni.objectType = VK_OBJECT_TYPE_IMAGE;
    ni.objectHandle = (uint64_t)m_image;
    ni.pObjectName = desc.debugName.c_str();
    auto fn = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
        m_device->vkDevice(), "vkSetDebugUtilsObjectNameEXT");
    if (fn) fn(m_device->vkDevice(), &ni);
  }
}

// Swapchain-image ctor: wraps external VkImage + view (no VMA allocation).
VulkanImage::VulkanImage(VulkanDevice* device, VkImage image, VkImageView view,
                         const ImageDesc& desc)
    : m_device(device), m_desc(desc), m_image(image), m_view(view),
      m_allocation(VK_NULL_HANDLE),
      m_aspectMask(VK_IMAGE_ASPECT_COLOR_BIT),
      m_ownsImage(false) {
}

VulkanImage::~VulkanImage() {
  // Always destroy the view (we created it).
  if (m_view != VK_NULL_HANDLE) {
    vkDestroyImageView(m_device->vkDevice(), m_view, nullptr);
    m_view = VK_NULL_HANDLE;
  }
  // Only destroy the VkImage if we own it (VMA-allocated).
  if (m_ownsImage && m_image != VK_NULL_HANDLE) {
    vmaDestroyImage(m_device->vmaAllocator(), m_image, m_allocation);
    m_image = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
  }
}

void VulkanImage::destroy() noexcept {
  if (!m_device->frameLoopActive()) {
    // Swapchain images: skip vkDeviceWaitIdle since the swapchain
    // lifecycle already ensures safety.
    if (m_ownsImage) {
      vkDeviceWaitIdle(m_device->vkDevice());
    }
    delete this;
    return;
  }
  delete this;  // Tier 1 deferred destroy belongs to R8.
}

// ---- VulkanSampler ---------------------------------------------------------
VulkanSampler::VulkanSampler(VulkanDevice* device, const SamplerDesc& desc)
    : m_device(device) {
  VkSamplerCreateInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  si.magFilter = toVkFilter(desc.magFilter);
  si.minFilter = toVkFilter(desc.minFilter);
  si.mipmapMode = (desc.minFilter == SamplerDesc::Filter::Linear)
                      ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                      : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  si.addressModeU = toVkAddressMode(desc.addressMode);
  si.addressModeV = toVkAddressMode(desc.addressMode);
  si.addressModeW = toVkAddressMode(desc.addressMode);
  si.maxLod = VK_LOD_CLAMP_NONE;
  si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

  VK_CHECK(vkCreateSampler(m_device->vkDevice(), &si, nullptr, &m_sampler));
}

VulkanSampler::~VulkanSampler() {
  if (m_sampler != VK_NULL_HANDLE) {
    vkDestroySampler(m_device->vkDevice(), m_sampler, nullptr);
    m_sampler = VK_NULL_HANDLE;
  }
}

void VulkanSampler::destroy() noexcept {
  if (!m_device->frameLoopActive()) {
    vkDeviceWaitIdle(m_device->vkDevice());
    delete this;
    return;
  }
  delete this;
}

}  // namespace ksk::rhi::vulkan
