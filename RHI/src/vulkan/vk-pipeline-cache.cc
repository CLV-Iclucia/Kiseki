//
// vk-pipeline-cache.cc
//

#include "vk-pipeline-cache.h"

#include "vk-internals.h"

namespace ksk::rhi::vulkan {

VulkanPipelineCache::VulkanPipelineCache(VkDevice device) : m_device(device) {
  VkPipelineCacheCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  ci.initialDataSize = 0;
  ci.pInitialData = nullptr;
  VK_CHECK(vkCreatePipelineCache(m_device, &ci, nullptr, &m_cache));
}

VulkanPipelineCache::~VulkanPipelineCache() {
  // Clear maps FIRST to release all PipelineRef/ShaderRef before destroying
  // the VkPipelineCache and before VkDevice is destroyed. This ensures
  // VulkanPipeline::destroy() and VulkanShader::destroy() run while the
  // device is still valid.
  m_graphics.clear();
  m_compute.clear();

  if (m_cache != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(m_device, m_cache, nullptr);
    m_cache = VK_NULL_HANDLE;
  }
}

PipelineRef VulkanPipelineCache::findCompute(
    const ComputePipelineDesc& desc) const {
  auto it = m_compute.find(desc);
  if (it != m_compute.end()) return it->second;
  return {};
}

PipelineRef VulkanPipelineCache::findGraphics(
    const GraphicsPipelineDesc& desc) const {
  auto it = m_graphics.find(desc);
  if (it != m_graphics.end()) return it->second;
  return {};
}

void VulkanPipelineCache::insertCompute(const ComputePipelineDesc& desc,
                                        PipelineRef pipeline) {
  m_compute[desc] = std::move(pipeline);
}

void VulkanPipelineCache::insertGraphics(const GraphicsPipelineDesc& desc,
                                         PipelineRef pipeline) {
  m_graphics[desc] = std::move(pipeline);
}

}  // namespace ksk::rhi::vulkan
