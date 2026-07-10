//
// vk-pipeline.cc
// Builds VkDescriptorSetLayouts + VkPipelineLayout + VkPipeline from
// reflected shader(s). Layout derivation is the heart of the "user never
// writes a layout" promise (plan §3.4 R1).
//
// R6: supports both compute (single shader) and graphics (VS + FS) with
// merged reflection for multi-stage layout derivation.
//

#include "vk-pipeline.h"

#include "vk-device.h"
#include "vk-internals.h"
#include "vk-shader.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ksk::rhi::vulkan {

namespace {

// Key for a (set, binding) pair used during merging.
struct SetBindingKey {
  uint32_t set;
  uint32_t binding;
  bool operator==(const SetBindingKey& o) const {
    return set == o.set && binding == o.binding;
  }
};

struct SetBindingKeyHash {
  size_t operator()(const SetBindingKey& k) const {
    return std::hash<uint64_t>{}(
        (static_cast<uint64_t>(k.set) << 32) | k.binding);
  }
};

// Per-set arrays of VkDescriptorSetLayoutBinding.
struct SetBuckets {
  std::array<std::vector<VkDescriptorSetLayoutBinding>, kMaxDescriptorSets>
      buckets{};
  uint32_t maxSet = 0;
  bool sawAny = false;
};

SetBuckets gatherBindings(const ReflectionInfo& ri, VkShaderStageFlags stages) {
  SetBuckets out;
  for (const auto& b : ri.bindings) {
    if (b.set >= kMaxDescriptorSets) {
      spdlog::error(
          "[VulkanPipeline] reflection lists set={} which exceeds RHI limit "
          "(kMaxDescriptorSets={}). Recompile the kernel against a lower set "
          "or raise the limit.",
          b.set, kMaxDescriptorSets);
      throw std::runtime_error("set index exceeds kMaxDescriptorSets");
    }
    VkDescriptorSetLayoutBinding lb{};
    lb.binding = b.binding;
    lb.descriptorType = toVkDescriptorType(b.type);
    lb.descriptorCount = b.count;
    lb.stageFlags = stages;
    out.buckets[b.set].push_back(lb);
    out.maxSet = std::max(out.maxSet, b.set);
    out.sawAny = true;
  }
  return out;
}

// Merge two SetBuckets: same (set, binding) -> stageFlags |= new stage.
SetBuckets mergeSetBuckets(const SetBuckets& a, const SetBuckets& b) {
  SetBuckets merged = a;
  merged.maxSet = std::max(a.maxSet, b.maxSet);
  merged.sawAny = a.sawAny || b.sawAny;

  for (uint32_t s = 0; s <= b.maxSet && s < kMaxDescriptorSets; ++s) {
    for (const auto& newBinding : b.buckets[s]) {
      bool found = false;
      for (auto& existing : merged.buckets[s]) {
        if (existing.binding == newBinding.binding) {
          if (existing.descriptorType != newBinding.descriptorType) {
            throw std::runtime_error(
                "VS and FS disagree on descriptor type at set=" +
                std::to_string(s) + " binding=" +
                std::to_string(newBinding.binding));
          }
          existing.stageFlags |= newBinding.stageFlags;
          existing.descriptorCount =
              std::max(existing.descriptorCount, newBinding.descriptorCount);
          found = true;
          break;
        }
      }
      if (!found) {
        merged.buckets[s].push_back(newBinding);
      }
    }
  }
  return merged;
}

}  // namespace

// ---- Compute pipeline ctor -------------------------------------------------
VulkanPipeline::VulkanPipeline(VulkanDevice* device,
                               const ComputePipelineDesc& desc)
    : m_device(device), m_shader(desc.shader) {
  if (!m_shader) {
    throw std::runtime_error(
        "ComputePipelineDesc::shader is null - cannot build pipeline");
  }
  if (m_shader->stage() != ShaderStage::Compute) {
    throw std::runtime_error(
        "createComputePipeline received a non-compute shader stage");
  }

  const ReflectionInfo* reflPtrs[] = {&m_shader->reflection()};
  buildLayout(reflPtrs, VK_SHADER_STAGE_COMPUTE_BIT);
  buildComputePipeline(desc);
}

// ---- Graphics pipeline ctor ------------------------------------------------
VulkanPipeline::VulkanPipeline(VulkanDevice* device,
                               const GraphicsPipelineDesc& desc)
    : m_device(device) {
  if (!desc.vertexShader || !desc.fragmentShader) {
    throw std::runtime_error(
        "GraphicsPipelineDesc: both vertexShader and fragmentShader required");
  }

  m_shader = desc.vertexShader;
  m_fragmentShader = desc.fragmentShader;
  m_bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

  // Merge VS + FS reflections for SHADER_PARAMS resolution
  m_mergedReflection =
      mergeReflections(m_shader->reflection(), m_fragmentShader->reflection());

  const ReflectionInfo* reflPtrs[] = {&m_shader->reflection(),
                                       &m_fragmentShader->reflection()};
  VkShaderStageFlags combinedStages =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  buildLayout(reflPtrs, combinedStages);
  buildGraphicsPipeline(desc);
}

VulkanPipeline::~VulkanPipeline() = default;

// ---- Multi-stage layout derivation -----------------------------------------
void VulkanPipeline::buildLayout(
    std::span<const ReflectionInfo* const> stages,
    VkShaderStageFlags combinedStages) {
  // Gather and merge bindings from all stages.
  // For simplicity, we use combinedStages for all bindings (Vulkan allows
  // broader stageFlags than strictly needed - harmless).
  SetBuckets merged{};
  for (const auto* ri : stages) {
    auto buckets = gatherBindings(*ri, combinedStages);
    if (!merged.sawAny) {
      merged = buckets;
    } else {
      merged = mergeSetBuckets(merged, buckets);
    }
  }

  // ---- Per-set descriptor set layouts -------------------------------------
  uint32_t setCount = merged.sawAny ? (merged.maxSet + 1) : 0;

  for (uint32_t i = 0; i < setCount; ++i) {
    VkDescriptorSetLayoutCreateInfo li{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = static_cast<uint32_t>(merged.buckets[i].size());
    li.pBindings = merged.buckets[i].data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->vkDevice(), &li, nullptr,
                                         &m_setLayouts[i]));
  }
  m_setLayoutCount = setCount;

  // ---- Push constant range -----------------------------------------------
  const ReflectionInfo& effectiveRefl =
      m_mergedReflection ? *m_mergedReflection
                         : m_shader->reflection();

  VkPushConstantRange pcRange{};
  bool hasPc = false;
  if (!effectiveRefl.pushConstants.empty()) {
    uint32_t lo = effectiveRefl.pushConstants[0].offset;
    uint32_t hi =
        effectiveRefl.pushConstants[0].offset + effectiveRefl.pushConstants[0].size;
    for (size_t i = 1; i < effectiveRefl.pushConstants.size(); ++i) {
      lo = std::min(lo, effectiveRefl.pushConstants[i].offset);
      hi = std::max(hi, effectiveRefl.pushConstants[i].offset +
                            effectiveRefl.pushConstants[i].size);
    }
    pcRange.stageFlags = combinedStages;
    pcRange.offset = lo;
    pcRange.size = hi - lo;
    hasPc = true;
    m_pushStages = combinedStages;
  }

  // ---- VkPipelineLayout ---------------------------------------------------
  VkPipelineLayoutCreateInfo li{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = setCount;
  li.pSetLayouts = setCount ? m_setLayouts.data() : nullptr;
  li.pushConstantRangeCount = hasPc ? 1u : 0u;
  li.pPushConstantRanges = hasPc ? &pcRange : nullptr;
  VK_CHECK(vkCreatePipelineLayout(m_device->vkDevice(), &li, nullptr,
                                  &m_layout));
}

// ---- Reflection merge (VS U FS) -------------------------------------------
ReflectionInfo VulkanPipeline::mergeReflections(const ReflectionInfo& vs,
                                                 const ReflectionInfo& fs) {
  ReflectionInfo merged;

  // Merge bindings: union by (set, binding). Same slot must have same type.
  std::unordered_map<SetBindingKey, DescriptorBindingInfo, SetBindingKeyHash>
      seen;

  for (const auto& b : vs.bindings) {
    SetBindingKey key{b.set, b.binding};
    seen[key] = b;
  }
  for (const auto& b : fs.bindings) {
    SetBindingKey key{b.set, b.binding};
    auto it = seen.find(key);
    if (it != seen.end()) {
      if (it->second.type != b.type) {
        throw std::runtime_error(
            "mergeReflections: VS and FS disagree on descriptor type at set=" +
            std::to_string(b.set) + " binding=" + std::to_string(b.binding));
      }
      it->second.count = std::max(it->second.count, b.count);
    } else {
      seen[key] = b;
    }
  }

  merged.bindings.reserve(seen.size());
  for (const auto& [_, info] : seen) {
    merged.bindings.push_back(info);
  }
  std::sort(merged.bindings.begin(), merged.bindings.end(),
            [](const DescriptorBindingInfo& a, const DescriptorBindingInfo& b) {
              return std::tie(a.set, a.binding) < std::tie(b.set, b.binding);
            });

  // Merge push constants: collect from both stages
  merged.pushConstants = vs.pushConstants;
  for (const auto& pc : fs.pushConstants) {
    merged.pushConstants.push_back(pc);
  }

  return merged;
}

// ---- Compute pipeline creation ---------------------------------------------
void VulkanPipeline::buildComputePipeline(const ComputePipelineDesc& desc) {
  auto* vkShader = static_cast<VulkanShader*>(desc.shader.get());

  VkPipelineShaderStageCreateInfo ss{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ss.module = vkShader->vkHandle();
  static thread_local std::string s_entryName;
  s_entryName.assign(vkShader->entryPoint());
  ss.pName = s_entryName.c_str();

  VkComputePipelineCreateInfo ci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = ss;
  ci.layout = m_layout;

  VK_CHECK(vkCreateComputePipelines(m_device->vkDevice(), VK_NULL_HANDLE, 1,
                                    &ci, nullptr, &m_pipeline));
}

// ---- Graphics pipeline creation --------------------------------------------
void VulkanPipeline::buildGraphicsPipeline(const GraphicsPipelineDesc& desc) {
  auto* vsShader = static_cast<VulkanShader*>(desc.vertexShader.get());
  auto* fsShader = static_cast<VulkanShader*>(desc.fragmentShader.get());

  // ---- Shader stages -------------------------------------------------------
  std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};
  std::array<std::string, 2> entryNames;

  for (size_t i = 0; i < 2; ++i) {
    auto* shader = i == 0 ? vsShader : fsShader;
    shaderStages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStages[i].stage =
        i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[i].module = shader->vkHandle();
    entryNames[i].assign(shader->entryPoint());
    shaderStages[i].pName = entryNames[i].c_str();
  }

  // ---- Vertex input state --------------------------------------------------
  std::vector<VkVertexInputBindingDescription> vkBindings;
  std::vector<VkVertexInputAttributeDescription> vkAttributes;

  for (const auto& binding : desc.vertexBindings) {
    VkVertexInputBindingDescription vb{};
    vb.binding = binding.binding;
    vb.stride = binding.stride;
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vkBindings.push_back(vb);

    for (const auto& attr : binding.attributes) {
      VkVertexInputAttributeDescription va{};
      va.location = attr.location;
      va.binding = binding.binding;
      va.format = toVkFormat(attr.format);
      va.offset = attr.offset;
      vkAttributes.push_back(va);
    }
  }

  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount =
      static_cast<uint32_t>(vkBindings.size());
  vi.pVertexBindingDescriptions = vkBindings.data();
  vi.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(vkAttributes.size());
  vi.pVertexAttributeDescriptions = vkAttributes.data();

  // ---- Input assembly (topology) -------------------------------------------
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  switch (desc.topology) {
    case GraphicsPipelineDesc::PrimitiveTopology::TriangleList:
      ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      break;
    case GraphicsPipelineDesc::PrimitiveTopology::LineList:
      ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      break;
    case GraphicsPipelineDesc::PrimitiveTopology::PointList:
      ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      break;
  }
  ia.primitiveRestartEnable = VK_FALSE;

  // ---- Viewport & scissor state (dynamic) ---------------------------------
  VkPipelineViewportStateCreateInfo vp{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  vp.pViewports = nullptr;
  vp.pScissors = nullptr;

  // ---- Rasterization state ------------------------------------------------
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.depthClampEnable = VK_FALSE;
  rs.rasterizerDiscardEnable = VK_FALSE;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_BACK_BIT;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.depthBiasEnable = VK_FALSE;
  rs.lineWidth = 1.0f;

  // ---- Multisample state --------------------------------------------------
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  ms.sampleShadingEnable = VK_FALSE;

  // ---- Depth-stencil state ------------------------------------------------
  VkPipelineDepthStencilStateCreateInfo ds{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = desc.depthTest ? VK_TRUE : VK_FALSE;
  ds.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS;
  ds.depthBoundsTestEnable = VK_FALSE;
  ds.stencilTestEnable = VK_FALSE;

  // ---- Color blend state --------------------------------------------------
  std::vector<VkPipelineColorBlendAttachmentState> colorAttachments;
  for (size_t i = 0; i < desc.colorFormats.size(); ++i) {
    VkPipelineColorBlendAttachmentState ca{};
    if (desc.alphaBlend) {
      ca.blendEnable = VK_TRUE;
      ca.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      ca.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      ca.colorBlendOp = VK_BLEND_OP_ADD;
      ca.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ca.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      ca.alphaBlendOp = VK_BLEND_OP_ADD;
    } else {
      ca.blendEnable = VK_FALSE;
    }
    ca.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorAttachments.push_back(ca);
  }

  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.logicOpEnable = VK_FALSE;
  cb.attachmentCount = static_cast<uint32_t>(colorAttachments.size());
  cb.pAttachments = colorAttachments.data();

  // ---- Dynamic state (viewport + scissor) ---------------------------------
  std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dyn{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dyn.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dyn.pDynamicStates = dynamicStates.data();

  // ---- Rendering info (Vulkan 1.3 dynamic rendering) ----------------------
  std::vector<VkFormat> colorFormats;
  for (const auto& fmt : desc.colorFormats) {
    colorFormats.push_back(toVkFormat(fmt));
  }

  VkFormat depthFormat = desc.depthFormat != Format::Undefined
                             ? toVkFormat(desc.depthFormat)
                             : VK_FORMAT_UNDEFINED;

  VkPipelineRenderingCreateInfoKHR rendering{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
  rendering.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
  rendering.pColorAttachmentFormats = colorFormats.data();
  rendering.depthAttachmentFormat = depthFormat;

  // ---- Graphics pipeline creation -----------------------------------------
  VkGraphicsPipelineCreateInfo ci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  ci.stageCount = 2;
  ci.pStages = shaderStages.data();
  ci.pVertexInputState = &vi;
  ci.pInputAssemblyState = &ia;
  ci.pViewportState = &vp;
  ci.pRasterizationState = &rs;
  ci.pMultisampleState = &ms;
  ci.pDepthStencilState = &ds;
  ci.pColorBlendState = &cb;
  ci.pDynamicState = &dyn;
  ci.layout = m_layout;
  ci.pNext = &rendering;

  VK_CHECK(vkCreateGraphicsPipelines(m_device->vkDevice(), VK_NULL_HANDLE, 1,
                                     &ci, nullptr, &m_pipeline));
}

// ---- Destroy ---------------------------------------------------------------
void VulkanPipeline::destroy() noexcept {
  vkDeviceWaitIdle(m_device->vkDevice());

  if (m_pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(m_device->vkDevice(), m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;
  }
  if (m_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(m_device->vkDevice(), m_layout, nullptr);
    m_layout = VK_NULL_HANDLE;
  }
  for (auto& l : m_setLayouts) {
    if (l != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device->vkDevice(), l, nullptr);
      l = VK_NULL_HANDLE;
    }
  }
  delete this;
}

// ---- reflection() ----------------------------------------------------------
const ReflectionInfo& VulkanPipeline::reflection() const {
  if (m_mergedReflection) return *m_mergedReflection;
  return m_shader->reflection();
}

}  // namespace ksk::rhi::vulkan
