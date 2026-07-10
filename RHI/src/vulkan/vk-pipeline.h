//
// vk-pipeline.h
// Vulkan Pipeline + layout derivation from ReflectionInfo.
// See docs/rhi-plan.md §3.4.2 / §6.3.
//

#pragma once

#include <RHI/pipeline.h>
#include <RHI/reflection.h>

#include <vulkan/vulkan.h>

#include <array>
#include <optional>
#include <span>

namespace ksk::rhi::vulkan {

class VulkanDevice;

// Vulkan 1.3 guarantees maxBoundDescriptorSets >= 4. We allow up to 4 sets
// in R4 — fluid kernels comfortably fit in a single set, but reserving 4
// keeps SHADER_PARAMS users free to organise binding spaces.
inline constexpr uint32_t kMaxDescriptorSets = 4;

class VulkanPipeline final : public Pipeline {
 public:
  // Compute pipeline ctor. Throws std::runtime_error on backend failure.
  VulkanPipeline(VulkanDevice* device, const ComputePipelineDesc& desc);

  // Graphics pipeline ctor (R6). Throws std::runtime_error on backend failure.
  VulkanPipeline(VulkanDevice* device, const GraphicsPipelineDesc& desc);

  ~VulkanPipeline() override;

  // ---- Pipeline interface -------------------------------------------------
  const ReflectionInfo& reflection() const override;

  // ---- Backend-internal ---------------------------------------------------
  VkPipeline vkPipeline() const noexcept { return m_pipeline; }
  VkPipelineLayout vkPipelineLayout() const noexcept { return m_layout; }
  VkPipelineBindPoint bindPoint() const noexcept { return m_bindPoint; }

  // Per-set descriptor layout. Returns VK_NULL_HANDLE if `set` is unused by
  // the pipeline (allowed; CommandList ignores those during flush).
  VkDescriptorSetLayout setLayout(uint32_t set) const noexcept {
    return set < kMaxDescriptorSets ? m_setLayouts[set] : VK_NULL_HANDLE;
  }

  // Cached for CommandList::pushAt — knows which stages to push for.
  VkShaderStageFlags pushStageFlags() const noexcept { return m_pushStages; }

  const ReflectionInfo& shaderReflection() const noexcept {
    return m_mergedReflection ? *m_mergedReflection : m_shader->reflection();
  }

 protected:
  void destroy() noexcept override;

 private:
  void buildLayout(std::span<const ReflectionInfo* const> stages,
                   VkShaderStageFlags combinedStages);
  void buildComputePipeline(const ComputePipelineDesc& desc);
  void buildGraphicsPipeline(const GraphicsPipelineDesc& desc);

  // Merge VS + FS reflections into a single combined ReflectionInfo.
  static ReflectionInfo mergeReflections(const ReflectionInfo& vs,
                                         const ReflectionInfo& fs);

  VulkanDevice* m_device;
  ShaderRef m_shader;         // compute shader (or VS for graphics)
  ShaderRef m_fragmentShader; // non-null only for graphics pipelines

  // Merged reflection for SHADER_PARAMS resolution (graphics: VS∪FS)
  std::optional<ReflectionInfo> m_mergedReflection;

  std::array<VkDescriptorSetLayout, kMaxDescriptorSets> m_setLayouts{};
  uint32_t m_setLayoutCount = 0;  // count of non-null entries

  VkPipelineLayout m_layout = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;
  VkPipelineBindPoint m_bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
  VkShaderStageFlags m_pushStages = 0;
};

}  // namespace ksk::rhi::vulkan
