//
// vk-pipeline-cache.h
// Device-level pipeline cache — wraps VkPipelineCache + std::map deduplication.
// See docs/rhi-r6-plan.md §2d.
//

#pragma once

#include <RHI/pipeline.h>
#include <vulkan/vulkan.h>

#include <map>

namespace ksk::rhi::vulkan {

class VulkanPipelineCache {
 public:
  explicit VulkanPipelineCache(VkDevice device);
  ~VulkanPipelineCache();

  // Look up cached pipeline. Returns null if miss.
  PipelineRef findCompute(const ComputePipelineDesc& desc) const;
  PipelineRef findGraphics(const GraphicsPipelineDesc& desc) const;

  // Insert into cache after successful creation.
  void insertCompute(const ComputePipelineDesc& desc, PipelineRef pipeline);
  void insertGraphics(const GraphicsPipelineDesc& desc, PipelineRef pipeline);

  // Underlying Vulkan pipeline cache handle (passed to vkCreate*Pipelines).
  VkPipelineCache vkCache() const noexcept { return m_cache; }

 private:
  VkDevice m_device;
  VkPipelineCache m_cache = VK_NULL_HANDLE;
  std::map<ComputePipelineDesc, PipelineRef> m_compute;
  std::map<GraphicsPipelineDesc, PipelineRef> m_graphics;
};

}  // namespace ksk::rhi::vulkan
