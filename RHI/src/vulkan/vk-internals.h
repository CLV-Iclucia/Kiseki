//
// vk-internals.h
// Vulkan-specific helpers. Not exposed outside RHI/src/vulkan/.
//
// Includes:
//   - VK_CHECK macro
//   - Format / usage / stage / access / layout converters (plan §13.4–13.5)
//   - Aspect mask helpers
//

#pragma once

#include <Core/properties.h>
#include <RHI/buffer.h>
#include <RHI/commands.h>
#include <RHI/format.h>
#include <RHI/image.h>
#include <RHI/reflection.h>
#include <RHI/shader.h>
#include <RHI/sync.h>
#include <spdlog/spdlog.h>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>

namespace ksk::rhi::vulkan {

// ---- VK_CHECK --------------------------------------------------------------
#define VK_CHECK(expr)                                                       \
  do {                                                                       \
    VkResult _vk_result = (expr);                                            \
    if (_vk_result != VK_SUCCESS) {                                          \
      spdlog::error("VK_CHECK failed: {} at {}:{}", #expr, __FILE__,         \
                    __LINE__);                                               \
      throw std::runtime_error(std::string("Vulkan call failed: ") + #expr); \
    }                                                                        \
  } while (0)

// ---- Format mapping --------------------------------------------------------
inline VkFormat toVkFormat(Format f) noexcept {
  switch (f) {
    case Format::Undefined:        return VK_FORMAT_UNDEFINED;

    case Format::R8_UNorm:         return VK_FORMAT_R8_UNORM;
    case Format::R8_SNorm:         return VK_FORMAT_R8_SNORM;
    case Format::R8_UInt:          return VK_FORMAT_R8_UINT;
    case Format::R8_SInt:          return VK_FORMAT_R8_SINT;
    case Format::RG8_UNorm:        return VK_FORMAT_R8G8_UNORM;
    case Format::RG8_SNorm:        return VK_FORMAT_R8G8_SNORM;
    case Format::RG8_UInt:         return VK_FORMAT_R8G8_UINT;
    case Format::RG8_SInt:         return VK_FORMAT_R8G8_SINT;
    case Format::RGBA8_UNorm:      return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8_SNorm:      return VK_FORMAT_R8G8B8A8_SNORM;
    case Format::RGBA8_UInt:       return VK_FORMAT_R8G8B8A8_UINT;
    case Format::RGBA8_SInt:       return VK_FORMAT_R8G8B8A8_SINT;
    case Format::RGBA8_UNorm_sRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::BGRA8_UNorm:      return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA8_UNorm_sRGB: return VK_FORMAT_B8G8R8A8_SRGB;

    case Format::R16_UNorm:        return VK_FORMAT_R16_UNORM;
    case Format::R16_UInt:         return VK_FORMAT_R16_UINT;
    case Format::R16_SInt:         return VK_FORMAT_R16_SINT;
    case Format::R16_Float:        return VK_FORMAT_R16_SFLOAT;
    case Format::RG16_UNorm:       return VK_FORMAT_R16G16_UNORM;
    case Format::RG16_UInt:        return VK_FORMAT_R16G16_UINT;
    case Format::RG16_SInt:        return VK_FORMAT_R16G16_SINT;
    case Format::RG16_Float:       return VK_FORMAT_R16G16_SFLOAT;
    case Format::RGBA16_UNorm:     return VK_FORMAT_R16G16B16A16_UNORM;
    case Format::RGBA16_UInt:      return VK_FORMAT_R16G16B16A16_UINT;
    case Format::RGBA16_SInt:      return VK_FORMAT_R16G16B16A16_SINT;
    case Format::RGBA16_Float:     return VK_FORMAT_R16G16B16A16_SFLOAT;

    case Format::R32_UInt:         return VK_FORMAT_R32_UINT;
    case Format::R32_SInt:         return VK_FORMAT_R32_SINT;
    case Format::R32_Float:        return VK_FORMAT_R32_SFLOAT;
    case Format::RG32_UInt:        return VK_FORMAT_R32G32_UINT;
    case Format::RG32_SInt:        return VK_FORMAT_R32G32_SINT;
    case Format::RG32_Float:       return VK_FORMAT_R32G32_SFLOAT;
    case Format::RGB32_Float:      return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::RGBA32_UInt:      return VK_FORMAT_R32G32B32A32_UINT;
    case Format::RGBA32_SInt:      return VK_FORMAT_R32G32B32A32_SINT;
    case Format::RGBA32_Float:     return VK_FORMAT_R32G32B32A32_SFLOAT;

    case Format::R10G10B10A2_UNorm: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::R11G11B10_Float:   return VK_FORMAT_B10G11R11_UFLOAT_PACK32;

    case Format::D16_UNorm:         return VK_FORMAT_D16_UNORM;
    case Format::D24_UNorm_S8_UInt: return VK_FORMAT_D24_UNORM_S8_UINT;
    case Format::D32_Float:         return VK_FORMAT_D32_SFLOAT;
    case Format::D32_Float_S8_UInt: return VK_FORMAT_D32_SFLOAT_S8_UINT;

    case Format::BC1_RGBA_UNorm:    return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case Format::BC3_RGBA_UNorm:    return VK_FORMAT_BC3_UNORM_BLOCK;
    case Format::BC5_RG_UNorm:      return VK_FORMAT_BC5_UNORM_BLOCK;
    case Format::BC7_RGBA_UNorm:    return VK_FORMAT_BC7_UNORM_BLOCK;
  }
  return VK_FORMAT_UNDEFINED;
}

// ---- Buffer / image usage --------------------------------------------------
inline VkBufferUsageFlags toVkBufferUsage(uint32_t mask) noexcept {
  VkBufferUsageFlags r = 0;
  if (mask & BufferDesc::Storage)     r |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (mask & BufferDesc::Uniform)     r |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (mask & BufferDesc::Vertex)      r |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (mask & BufferDesc::Index)       r |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (mask & BufferDesc::Indirect)    r |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (mask & BufferDesc::TransferSrc) r |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (mask & BufferDesc::TransferDst) r |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  return r;
}

inline VkImageUsageFlags toVkImageUsage(uint32_t mask) noexcept {
  VkImageUsageFlags r = 0;
  if (mask & ImageDesc::Storage)         r |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (mask & ImageDesc::Sampled)         r |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (mask & ImageDesc::ColorAttachment) r |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (mask & ImageDesc::DepthAttachment) r |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (mask & ImageDesc::TransferSrc)     r |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (mask & ImageDesc::TransferDst)     r |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  return r;
}

inline VkImageType toVkImageType(ImageDesc::Dim d) noexcept {
  switch (d) {
    case ImageDesc::Dim::D1: return VK_IMAGE_TYPE_1D;
    case ImageDesc::Dim::D2: return VK_IMAGE_TYPE_2D;
    case ImageDesc::Dim::D3: return VK_IMAGE_TYPE_3D;
  }
  return VK_IMAGE_TYPE_2D;
}

inline VkImageViewType toVkImageViewType(ImageDesc::Dim d, uint32_t arrayLayers) noexcept {
  switch (d) {
    case ImageDesc::Dim::D1:
      return arrayLayers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case ImageDesc::Dim::D2:
      return arrayLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    case ImageDesc::Dim::D3:
      return VK_IMAGE_VIEW_TYPE_3D;  // 3D arrays don't exist in Vulkan
  }
  return VK_IMAGE_VIEW_TYPE_2D;
}

// ---- VMA memory usage ------------------------------------------------------
inline VmaMemoryUsage toVmaMemoryUsage(BufferDesc::Visibility v) noexcept {
  switch (v) {
    case BufferDesc::Visibility::DeviceLocal: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case BufferDesc::Visibility::HostVisible: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    case BufferDesc::Visibility::Readback:    return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  }
  return VMA_MEMORY_USAGE_AUTO;
}

inline VmaAllocationCreateFlags toVmaAllocationFlags(BufferDesc::Visibility v) noexcept {
  switch (v) {
    case BufferDesc::Visibility::DeviceLocal: return 0;
    case BufferDesc::Visibility::HostVisible:
      return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
             VMA_ALLOCATION_CREATE_MAPPED_BIT;
    case BufferDesc::Visibility::Readback:
      return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
             VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  return 0;
}

// ---- Sampler mapping -------------------------------------------------------
inline VkFilter toVkFilter(SamplerDesc::Filter f) noexcept {
  switch (f) {
    case SamplerDesc::Filter::Nearest: return VK_FILTER_NEAREST;
    case SamplerDesc::Filter::Linear:  return VK_FILTER_LINEAR;
  }
  return VK_FILTER_LINEAR;
}

inline VkSamplerAddressMode toVkAddressMode(SamplerDesc::AddressMode m) noexcept {
  switch (m) {
    case SamplerDesc::AddressMode::Repeat:        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SamplerDesc::AddressMode::ClampToEdge:   return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case SamplerDesc::AddressMode::ClampToBorder: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  }
  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

// ---- Aspect mask -----------------------------------------------------------
inline VkImageAspectFlags aspectMaskFor(Format f) noexcept {
  if (formatHasStencil(f)) {
    return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  if (formatIsDepth(f)) {
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

// ---- Sync2 stage / access / layout ----------------------------------------
inline VkPipelineStageFlags2 toVkStage(uint32_t mask) noexcept {
  VkPipelineStageFlags2 r = 0;
  if (mask & BarrierDesc::StageComputeShader)        r |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  if (mask & BarrierDesc::StageVertexShader)         r |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
  if (mask & BarrierDesc::StageFragmentShader)       r |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  if (mask & BarrierDesc::StageTransfer)             r |= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  if (mask & BarrierDesc::StageAllGraphics)          r |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
  if (mask & BarrierDesc::StageTopOfPipe)            r |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  if (mask & BarrierDesc::StageBottomOfPipe)         r |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  if (mask & BarrierDesc::StageColorAttachmentOutput) r |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (mask & BarrierDesc::StageEarlyFragmentTests)   r |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
  if (mask & BarrierDesc::StageLateFragmentTests)    r |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  return r ? r : VK_PIPELINE_STAGE_2_NONE;
}

inline VkAccessFlags2 toVkAccess(uint32_t mask) noexcept {
  VkAccessFlags2 r = 0;
  if (mask & BarrierDesc::AccessShaderRead)           r |= VK_ACCESS_2_SHADER_READ_BIT;
  if (mask & BarrierDesc::AccessShaderWrite)          r |= VK_ACCESS_2_SHADER_WRITE_BIT;
  if (mask & BarrierDesc::AccessTransferRead)         r |= VK_ACCESS_2_TRANSFER_READ_BIT;
  if (mask & BarrierDesc::AccessTransferWrite)        r |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
  if (mask & BarrierDesc::AccessVertexRead)           r |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
  if (mask & BarrierDesc::AccessIndexRead)            r |= VK_ACCESS_2_INDEX_READ_BIT;
  if (mask & BarrierDesc::AccessUniformRead)          r |= VK_ACCESS_2_UNIFORM_READ_BIT;
  if (mask & BarrierDesc::AccessColorAttachmentRead)  r |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
  if (mask & BarrierDesc::AccessColorAttachmentWrite) r |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  if (mask & BarrierDesc::AccessDepthStencilRead)     r |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
  if (mask & BarrierDesc::AccessDepthStencilWrite)    r |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  return r;
}

inline VkImageLayout toVkImageLayout(BarrierDesc::ImageBarrier::Layout l) noexcept {
  using L = BarrierDesc::ImageBarrier::Layout;
  switch (l) {
    case L::Undefined:       return VK_IMAGE_LAYOUT_UNDEFINED;
    case L::General:         return VK_IMAGE_LAYOUT_GENERAL;
    case L::ShaderReadOnly:  return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case L::TransferSrc:     return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case L::TransferDst:     return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case L::ColorAttachment: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case L::DepthAttachment: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case L::Present:         return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  }
  return VK_IMAGE_LAYOUT_UNDEFINED;
}

// ---- R4 / R5: descriptor + shader stage mapping ---------------------------

// Map our backend-neutral descriptor type to VkDescriptorType. Combined
// image samplers are emitted by HLSL when a Texture+Sampler is referenced
// together; SampledImage stays as VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE so the
// CommandList::bindImageAt path with a non-null sampler upgrades it to
// COMBINED_IMAGE_SAMPLER on the fly via vk-commands.cc's flush path.
inline VkDescriptorType toVkDescriptorType(DescriptorBindingInfo::Type t) noexcept {
  using DT = DescriptorBindingInfo::Type;
  switch (t) {
    case DT::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case DT::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case DT::StorageImage:  return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case DT::SampledImage:  return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case DT::Sampler:       return VK_DESCRIPTOR_TYPE_SAMPLER;
  }
  return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

inline VkShaderStageFlags toVkShaderStage(ShaderStage s) noexcept {
  switch (s) {
    case ShaderStage::Compute:  return VK_SHADER_STAGE_COMPUTE_BIT;
    case ShaderStage::Vertex:   return VK_SHADER_STAGE_VERTEX_BIT;
    case ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  return 0;
}

}  // namespace ksk::rhi::vulkan
