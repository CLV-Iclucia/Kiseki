//
// vk-swapchain.cc
// Vulkan swapchain implementation.
// See docs/rhi-r7-swapchain-plan.md.
//

#include "vk-swapchain.h"

#include "vk-device.h"
#include "vk-image.h"
#include "vk-internals.h"
#include "vk-sync.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace ksk::rhi::vulkan {

// ============================================================================
// VulkanSwapchain
// ============================================================================

VulkanSwapchain::VulkanSwapchain(VulkanDevice* device,
                                 const SwapchainDesc& desc)
    : m_device(device),
      m_width(desc.width),
      m_height(desc.height),
      m_desiredImageCount(desc.imageCount),
      m_desiredPresentMode(desc.presentMode) {
  // ---- 1. Create surface via user callback ---------------------------------
  if (!desc.surfaceCreator) {
    throw std::runtime_error(
        "Rhi: SwapchainDesc::surfaceCreator must be provided");
  }

  void* surfRaw = desc.surfaceCreator(
      reinterpret_cast<void*>(m_device->vkInstance()));
  if (!surfRaw) {
    throw std::runtime_error(
        "Rhi: surfaceCreator callback returned nullptr -- "
        "surface creation failed");
  }
  m_surface = reinterpret_cast<VkSurfaceKHR>(surfRaw);

  // ---- 2. Verify present support on graphics queue -------------------------
  VkBool32 presentSupported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(
      m_device->vkPhysicalDevice(),
      m_device->queueFamilyForType(QueueType::Graphics),
      m_surface, &presentSupported);

  if (!presentSupported) {
    vkDestroySurfaceKHR(m_device->vkInstance(), m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
    throw std::runtime_error(
        "Rhi: graphics queue family does not support presentation to "
        "the created surface");
  }

  m_presentQueue = m_device->queueForType(QueueType::Graphics);

  // ---- 3. Build the swapchain ----------------------------------------------
  build();

  spdlog::info("Rhi: swapchain created ({}x{}, {} images, format {})",
               m_width, m_height, m_images.size(),
               static_cast<uint32_t>(m_format));
}

VulkanSwapchain::~VulkanSwapchain() {
  // Ensure GPU idle before teardown -- defensive, caller should have
  // already waited. This matches the Tier 0 philosophy.
  vkDeviceWaitIdle(m_device->vkDevice());

  destroyBackbuffers();

  if (m_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(m_device->vkDevice(), m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
  }
  if (m_surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(m_device->vkInstance(), m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }
}

// ---- acquireNextImage ------------------------------------------------------
ImageRef VulkanSwapchain::acquireNextImage(Semaphore& signalSemaphore) {
  auto* sem = static_cast<VulkanSemaphore*>(&signalSemaphore);

  uint32_t imageIdx = 0;
  VkResult result = vkAcquireNextImageKHR(
      m_device->vkDevice(), m_swapchain, UINT64_MAX,
      sem->vkHandle(), VK_NULL_HANDLE, &imageIdx);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    // Swapchain is out of date -- caller should recreate.
    return ImageRef{};
  }
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    spdlog::error("Rhi: vkAcquireNextImageKHR failed (VkResult {})",
                  static_cast<int>(result));
    return ImageRef{};
  }

  m_currentImageIdx = imageIdx;
  return m_images[imageIdx];
}

// ---- present ---------------------------------------------------------------
bool VulkanSwapchain::present(Semaphore& waitSemaphore) {
  auto* sem = static_cast<VulkanSemaphore*>(&waitSemaphore);
  VkSemaphore waitHandle = sem->vkHandle();

  VkPresentInfoKHR pi{};
  pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &waitHandle;
  pi.swapchainCount = 1;
  pi.pSwapchains = &m_swapchain;
  pi.pImageIndices = &m_currentImageIdx;

  VkResult result = vkQueuePresentKHR(m_presentQueue, &pi);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    return false;  // Caller should recreate.
  }
  if (result != VK_SUCCESS) {
    spdlog::error("Rhi: vkQueuePresentKHR failed (VkResult {})",
                  static_cast<int>(result));
    return false;
  }
  return true;
}

// ---- recreate --------------------------------------------------------------
void VulkanSwapchain::recreate(uint32_t newWidth, uint32_t newHeight) {
  // Safety net: wait for all GPU work. The caller should have already done
  // this (via waitFence on all frame slots), but belt-and-suspenders.
  vkDeviceWaitIdle(m_device->vkDevice());

  m_width = newWidth;
  m_height = newHeight;

  destroyBackbuffers();
  // Don't destroy old m_swapchain here -- pass it as oldSwapchain to
  // vkCreateSwapchainKHR in build() for driver-level recycling.
  build();

  spdlog::info("Rhi: swapchain recreated ({}x{}, {} images)",
               m_width, m_height, m_images.size());
}

// ---- build -----------------------------------------------------------------
void VulkanSwapchain::build() {
  VkPhysicalDevice physDev = m_device->vkPhysicalDevice();

  // Query surface capabilities.
  VkSurfaceCapabilitiesKHR caps{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, m_surface, &caps));

  // Clamp extent to capabilities.
  if (caps.currentExtent.width != 0xFFFFFFFF) {
    m_width = caps.currentExtent.width;
    m_height = caps.currentExtent.height;
  } else {
    m_width = std::clamp(m_width, caps.minImageExtent.width,
                         caps.maxImageExtent.width);
    m_height = std::clamp(m_height, caps.minImageExtent.height,
                          caps.maxImageExtent.height);
  }

  // Clamp image count.
  uint32_t imgCount = std::max(m_desiredImageCount, caps.minImageCount);
  if (caps.maxImageCount > 0) {
    imgCount = std::min(imgCount, caps.maxImageCount);
  }

  // Choose surface format.
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, m_surface, &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, m_surface, &formatCount,
                                       formats.data());
  VkSurfaceFormatKHR surfFormat = chooseSurfaceFormat(formats, m_format);

  // Choose present mode.
  uint32_t modeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, m_surface, &modeCount,
                                            nullptr);
  std::vector<VkPresentModeKHR> modes(modeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, m_surface, &modeCount,
                                            modes.data());
  VkPresentModeKHR presentMode = choosePresentMode(modes, m_desiredPresentMode);

  // Pre-transform: use current transform (no rotation).
  VkSurfaceTransformFlagBitsKHR preTransform = caps.currentTransform;

  // Composite alpha: prefer opaque.
  VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  if (!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
    compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  }

  // Create swapchain.
  VkSwapchainCreateInfoKHR ci{};
  ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  ci.surface = m_surface;
  ci.minImageCount = imgCount;
  ci.imageFormat = surfFormat.format;
  ci.imageColorSpace = surfFormat.colorSpace;
  ci.imageExtent = {m_width, m_height};
  ci.imageArrayLayers = 1;
  ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // for clears / blits
  ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.preTransform = preTransform;
  ci.compositeAlpha = compositeAlpha;
  ci.presentMode = presentMode;
  ci.clipped = VK_TRUE;
  ci.oldSwapchain = m_swapchain;  // VK_NULL_HANDLE on first build

  VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
  VK_CHECK(vkCreateSwapchainKHR(m_device->vkDevice(), &ci, nullptr,
                                &newSwapchain));

  // If we had an old swapchain (recreate path), destroy it now.
  if (m_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(m_device->vkDevice(), m_swapchain, nullptr);
  }
  m_swapchain = newSwapchain;

  // Store the actual format chosen.
  m_format = fromVkFormat(surfFormat.format);

  // ---- Retrieve backbuffer VkImages and wrap in VulkanImage (non-owning) ----
  uint32_t actualCount = 0;
  vkGetSwapchainImagesKHR(m_device->vkDevice(), m_swapchain, &actualCount,
                          nullptr);
  std::vector<VkImage> vkImages(actualCount);
  vkGetSwapchainImagesKHR(m_device->vkDevice(), m_swapchain, &actualCount,
                          vkImages.data());

  m_images.clear();
  m_images.reserve(actualCount);

  for (uint32_t i = 0; i < actualCount; ++i) {
    // Create per-image VkImageView.
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = vkImages[i];
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = surfFormat.format;
    vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY};
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(m_device->vkDevice(), &vi, nullptr, &view));

    ImageDesc imgDesc{};
    imgDesc.dim = ImageDesc::Dim::D2;
    imgDesc.width = m_width;
    imgDesc.height = m_height;
    imgDesc.depth = 1;
    imgDesc.format = m_format;
    imgDesc.mipLevels = 1;
    imgDesc.arrayLayers = 1;
    imgDesc.usage = ImageDesc::ColorAttachment | ImageDesc::TransferDst;
    imgDesc.debugName = "swapchain_image_" + std::to_string(i);

    // VulkanImage in non-owning mode: wraps the swapchain's VkImage
    // but does NOT call vmaDestroyImage on destruction.
    m_images.push_back(
        ImageRef(new VulkanImage(m_device, vkImages[i], view, imgDesc)));
  }
}

// ---- destroyBackbuffers ----------------------------------------------------
void VulkanSwapchain::destroyBackbuffers() {
  // Release our ImageRefs. Since VulkanImage is in non-owning mode,
  // its destructor destroys only the VkImageView, NOT the VkImage
  // (which belongs to the VkSwapchainKHR).
  m_images.clear();
}

// ---- chooseSurfaceFormat ---------------------------------------------------
VkSurfaceFormatKHR VulkanSwapchain::chooseSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available, Format preferred) {
  VkFormat vkPreferred = toVkFormat(preferred);

  // Try exact match.
  for (const auto& f : available) {
    if (f.format == vkPreferred) return f;
  }

  // Fallback: prefer BGRA8 sRGB (common desktop default).
  for (const auto& f : available) {
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return f;
    }
  }

  // Fallback: prefer BGRA8 UNORM.
  for (const auto& f : available) {
    if (f.format == VK_FORMAT_B8G8R8A8_UNORM) return f;
  }

  // Last resort: first available.
  if (!available.empty()) return available[0];

  // Should not happen -- surface with zero formats is broken.
  return {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
}

// ---- choosePresentMode -----------------------------------------------------
VkPresentModeKHR VulkanSwapchain::choosePresentMode(
    const std::vector<VkPresentModeKHR>& available, PresentMode preferred) {
  VkPresentModeKHR target;
  switch (preferred) {
    case PresentMode::Mailbox:   target = VK_PRESENT_MODE_MAILBOX_KHR; break;
    case PresentMode::Immediate: target = VK_PRESENT_MODE_IMMEDIATE_KHR; break;
    case PresentMode::Fifo:
    default:                     target = VK_PRESENT_MODE_FIFO_KHR; break;
  }

  for (auto m : available) {
    if (m == target) return target;
  }

  // FIFO is guaranteed by spec to always be available.
  return VK_PRESENT_MODE_FIFO_KHR;
}

// ---- fromVkFormat (reverse mapping, swapchain-internal only) ----------------
Format VulkanSwapchain::fromVkFormat(VkFormat f) {
  switch (f) {
    case VK_FORMAT_B8G8R8A8_SRGB:   return Format::BGRA8_UNorm_sRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:  return Format::BGRA8_UNorm;
    case VK_FORMAT_R8G8B8A8_SRGB:   return Format::RGBA8_UNorm_sRGB;
    case VK_FORMAT_R8G8B8A8_UNORM:  return Format::RGBA8_UNorm;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return Format::R10G10B10A2_UNorm;
    default:
      spdlog::warn("Rhi: unmapped VkFormat {} in swapchain; treating as BGRA8_UNorm_sRGB",
                   static_cast<int>(f));
      return Format::BGRA8_UNorm_sRGB;
  }
}

}  // namespace ksk::rhi::vulkan
