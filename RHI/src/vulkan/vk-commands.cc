//
// vk-commands.cc
// R0–R6 command list: transfer + barrier + clear + debug markers + compute + graphics.
//

#include "vk-commands.h"

#include "vk-buffer.h"
#include "vk-descriptor-allocator.h"
#include "vk-device.h"
#include "vk-image.h"
#include "vk-internals.h"
#include "vk-pipeline.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ksk::rhi::vulkan {

// ---- FNV-1a 64-bit hash helper for descriptor cache -------------------------
static uint64_t fnv1a(const void* data, size_t len, uint64_t seed = 0xcbf29ce484222325ULL) {
  auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = seed;
  for (size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

uint64_t VulkanCommandList::computeDescriptorContentHash(
    const VulkanPipeline* pipeline, uint32_t setIdx,
    const std::vector<PendingWrite>& writes) {
  uint64_t h = fnv1a(&pipeline, sizeof(pipeline));
  h = fnv1a(&setIdx, sizeof(setIdx), h);
  for (const auto& pw : writes) {
    h = fnv1a(&pw.binding, sizeof(pw.binding), h);
    h = fnv1a(&pw.type, sizeof(pw.type), h);
    h = fnv1a(&pw.kind, sizeof(pw.kind), h);
    if (pw.kind == PendingWrite::Kind::Buffer) {
      h = fnv1a(&pw.bufInfo, sizeof(pw.bufInfo), h);
    } else {
      h = fnv1a(&pw.imgInfo, sizeof(pw.imgInfo), h);
    }
  }
  return h;
}

VulkanCommandList::VulkanCommandList(VulkanDevice* device, QueueType queue,
                                     VkCommandBuffer cb)
    : m_device(device), m_queue(queue), m_cmd(cb) {}

VulkanCommandList::~VulkanCommandList() {
  // R0–R2 simplification: command buffers live in transient per-queue pools
  // owned by VulkanDevice. They are not individually freed here — the pool's
  // destruction at device teardown reclaims them. If the user dropped this
  // command list without submitting, end recording so the buffer is in a
  // valid state for pool destruction.
  endIfRecording();
  // (Future R8: free the buffer back to the pool eagerly when GPU is idle.)
}

void VulkanCommandList::endIfRecording() {
  if (m_recording) {
    VK_CHECK(vkEndCommandBuffer(m_cmd));
    m_recording = false;
  }
}

void VulkanCommandList::end() { endIfRecording(); }

// ---- Transfer --------------------------------------------------------------
void VulkanCommandList::copyBuffer(BufferRef src, BufferRef dst,
                                   std::span<const BufferCopy> regions) {
  std::vector<VkBufferCopy> rs;
  rs.reserve(regions.size());
  for (const auto& r : regions) {
    VkBufferCopy c{};
    c.srcOffset = r.srcOffset;
    c.dstOffset = r.dstOffset;
    c.size = r.size;
    rs.push_back(c);
  }
  vkCmdCopyBuffer(m_cmd,
                  static_cast<VulkanBuffer*>(src.get())->vkHandle(),
                  static_cast<VulkanBuffer*>(dst.get())->vkHandle(),
                  static_cast<uint32_t>(rs.size()), rs.data());
}

static VkBufferImageCopy toVkBufferImageCopy(const BufferImageCopy& r,
                                             VkImageAspectFlags aspect) {
  VkBufferImageCopy c{};
  c.bufferOffset = r.bufferOffset;
  c.bufferRowLength = r.bufferRowLength;
  c.bufferImageHeight = r.bufferImageHeight;
  c.imageSubresource.aspectMask = aspect;
  c.imageSubresource.mipLevel = r.mipLevel;
  c.imageSubresource.baseArrayLayer = r.baseArrayLayer;
  c.imageSubresource.layerCount = r.layerCount;
  c.imageOffset = {static_cast<int32_t>(r.imageOffsetX),
                   static_cast<int32_t>(r.imageOffsetY),
                   static_cast<int32_t>(r.imageOffsetZ)};
  c.imageExtent = {r.imageExtentW, r.imageExtentH, r.imageExtentD};
  return c;
}

void VulkanCommandList::copyBufferToImage(
    BufferRef src, ImageRef dst, std::span<const BufferImageCopy> regions) {
  auto* vimg = static_cast<VulkanImage*>(dst.get());
  std::vector<VkBufferImageCopy> rs;
  rs.reserve(regions.size());
  for (const auto& r : regions) {
    rs.push_back(toVkBufferImageCopy(r, vimg->aspectMask()));
  }
  vkCmdCopyBufferToImage(m_cmd,
                         static_cast<VulkanBuffer*>(src.get())->vkHandle(),
                         vimg->vkHandle(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         static_cast<uint32_t>(rs.size()), rs.data());
}

void VulkanCommandList::copyImageToBuffer(
    ImageRef src, BufferRef dst, std::span<const BufferImageCopy> regions) {
  auto* vimg = static_cast<VulkanImage*>(src.get());
  std::vector<VkBufferImageCopy> rs;
  rs.reserve(regions.size());
  for (const auto& r : regions) {
    rs.push_back(toVkBufferImageCopy(r, vimg->aspectMask()));
  }
  vkCmdCopyImageToBuffer(m_cmd, vimg->vkHandle(),
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         static_cast<VulkanBuffer*>(dst.get())->vkHandle(),
                         static_cast<uint32_t>(rs.size()), rs.data());
}

void VulkanCommandList::clearImage(ImageRef image, const ClearValue& value) {
  auto* vimg = static_cast<VulkanImage*>(image.get());

  VkImageSubresourceRange range{};
  range.aspectMask = vimg->aspectMask();
  range.baseMipLevel = 0;
  range.levelCount = VK_REMAINING_MIP_LEVELS;
  range.baseArrayLayer = 0;
  range.layerCount = VK_REMAINING_ARRAY_LAYERS;

  if (range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
    VkClearDepthStencilValue v{};
    v.depth = value.depthStencil.depth;
    v.stencil = value.depthStencil.stencil;
    vkCmdClearDepthStencilImage(m_cmd, vimg->vkHandle(),
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1,
                                &range);
  } else {
    VkClearColorValue v{};
    std::memcpy(&v.float32[0], value.colorF, sizeof(value.colorF));
    vkCmdClearColorImage(m_cmd, vimg->vkHandle(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
  }
}

void VulkanCommandList::fillBuffer(BufferRef buffer, uint32_t value) {
  auto* vbuf = static_cast<VulkanBuffer*>(buffer.get());
  vkCmdFillBuffer(m_cmd, vbuf->vkHandle(), 0, VK_WHOLE_SIZE, value);
}

// ---- Barrier (sync2 path) --------------------------------------------------
void VulkanCommandList::barrier(const BarrierDesc& desc) {
  std::vector<VkImageMemoryBarrier2> imgBarriers;
  imgBarriers.reserve(desc.imageBarriers.size());
  for (const auto& ib : desc.imageBarriers) {
    auto* vimg = static_cast<VulkanImage*>(ib.image.get());
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = toVkStage(desc.srcStage);
    b.srcAccessMask = toVkAccess(desc.srcAccess);
    b.dstStageMask = toVkStage(desc.dstStage);
    b.dstAccessMask = toVkAccess(desc.dstAccess);
    b.oldLayout = toVkImageLayout(ib.oldLayout);
    b.newLayout = toVkImageLayout(ib.newLayout);
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = vimg->vkHandle();
    b.subresourceRange.aspectMask = vimg->aspectMask();
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    imgBarriers.push_back(b);
  }

  VkMemoryBarrier2 mem{};
  mem.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
  mem.srcStageMask = toVkStage(desc.srcStage);
  mem.srcAccessMask = toVkAccess(desc.srcAccess);
  mem.dstStageMask = toVkStage(desc.dstStage);
  mem.dstAccessMask = toVkAccess(desc.dstAccess);

  VkDependencyInfo dep{};
  dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  if (imgBarriers.empty()) {
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mem;
  } else {
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(imgBarriers.size());
    dep.pImageMemoryBarriers = imgBarriers.data();
  }
  vkCmdPipelineBarrier2(m_cmd, &dep);
}

// ---- Debug markers ---------------------------------------------------------
void VulkanCommandList::debugMarkerBegin(std::string_view name) {
  if (!m_device->validationEnabled()) return;
  auto fn = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(
      m_device->vkDevice(), "vkCmdBeginDebugUtilsLabelEXT");
  if (!fn) return;
  std::string s(name);
  VkDebugUtilsLabelEXT label{};
  label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label.pLabelName = s.c_str();
  fn(m_cmd, &label);
  ++m_markerDepth;
}

void VulkanCommandList::debugMarkerEnd() {
  if (!m_device->validationEnabled()) return;
  if (m_markerDepth <= 0) return;
  auto fn = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(
      m_device->vkDevice(), "vkCmdEndDebugUtilsLabelEXT");
  if (!fn) return;
  fn(m_cmd);
  --m_markerDepth;
}

// ---- R4 / R5: compute -----------------------------------------------------
//
// State machine summary (plan §13.3):
//   bindComputePipeline  → vkCmdBindPipeline + clearPending
//   bind*At / pushAt     → accumulate into m_pendingSets / m_pendingPc
//   rawDispatch          → flushPending (vkUpdateDescriptorSets +
//                          vkCmdBindDescriptorSets + vkCmdPushConstants)
//                        → vkCmdDispatch
//
// We never call vkUpdate / vkCmdBind* in the bind*At entry points themselves
// — Vulkan's two-phase descriptor model demands batch flush.

void VulkanCommandList::clearPending() {
  for (auto& s : m_pendingSets) s.clear();
  m_pcDirtyLo = kVkCmdPushConstantArenaBytes;
  m_pcDirtyHi = 0;
}

void VulkanCommandList::bindComputePipeline(PipelineRef pipeline) {
  if (!pipeline) {
    throw std::runtime_error(
        "VulkanCommandList::bindComputePipeline received null pipeline");
  }
  auto* vkPso = static_cast<VulkanPipeline*>(pipeline.get());
  if (vkPso->vkPipeline() == m_boundPipelineHandle) {
    // Same PSO already bound; nothing to do.
    return;
  }
  vkCmdBindPipeline(m_cmd, vkPso->bindPoint(), vkPso->vkPipeline());

  m_boundPipeline = vkPso;
  m_boundPipelineHandle = vkPso->vkPipeline();
  m_boundLayout = vkPso->vkPipelineLayout();

  // Switching pipeline invalidates any pending writes — set layouts may
  // change between PSOs and the descriptor sets allocated against the new
  // layout would be wrong.
  clearPending();
}

VkDescriptorType VulkanCommandList::lookupDescriptorType(uint32_t set,
                                                         uint32_t binding) const {
  if (!m_boundPipeline) {
    throw std::runtime_error(
        "bind*At() called before bindComputePipeline — no reflection to "
        "consult");
  }
  const auto& ri = m_boundPipeline->shaderReflection();
  for (const auto& b : ri.bindings) {
    if (b.set == set && b.binding == binding) {
      return toVkDescriptorType(b.type);
    }
  }
  throw std::runtime_error(
      std::string("bind at (set=") + std::to_string(set) +
      ", binding=" + std::to_string(binding) +
      ") not found in current pipeline reflection");
}

void VulkanCommandList::bindBufferAt(uint32_t set, uint32_t binding,
                                     BufferRef buf) {
  if (set >= kVkCmdMaxDescriptorSets) {
    throw std::runtime_error("bindBufferAt: set index >= kVkCmdMaxDescriptorSets");
  }
  if (!buf) {
    throw std::runtime_error("bindBufferAt: null BufferRef");
  }

  PendingWrite w;
  w.binding = binding;
  w.type = lookupDescriptorType(set, binding);
  w.kind = PendingWrite::Kind::Buffer;
  w.bufInfo.buffer = static_cast<VulkanBuffer*>(buf.get())->vkHandle();
  w.bufInfo.offset = 0;
  w.bufInfo.range = VK_WHOLE_SIZE;

  m_pendingSets[set].writes.push_back(w);
  m_pendingSets[set].dirty = true;
}

void VulkanCommandList::bindImageAt(uint32_t set, uint32_t binding,
                                    ImageRef img, SamplerRef sampler) {
  if (set >= kVkCmdMaxDescriptorSets) {
    throw std::runtime_error("bindImageAt: set index >= kVkCmdMaxDescriptorSets");
  }
  if (!img) {
    throw std::runtime_error("bindImageAt: null ImageRef");
  }

  PendingWrite w;
  w.binding = binding;
  w.type = lookupDescriptorType(set, binding);
  w.kind = PendingWrite::Kind::Image;

  auto* vkImg = static_cast<VulkanImage*>(img.get());
  auto* vkSmp = sampler ? static_cast<VulkanSampler*>(sampler.get()) : nullptr;

  w.imgInfo.imageView = vkImg->vkView();
  w.imgInfo.sampler = vkSmp ? vkSmp->vkHandle() : VK_NULL_HANDLE;
  // Layout: STORAGE_IMAGE descriptors must be VK_IMAGE_LAYOUT_GENERAL;
  // SAMPLED / COMBINED descriptors expect SHADER_READ_ONLY_OPTIMAL. The
  // user is responsible for transitioning the image with `barrier()` before
  // dispatch — we just declare the layout we expect.
  if (w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
    w.imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  } else {
    w.imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }

  m_pendingSets[set].writes.push_back(w);
  m_pendingSets[set].dirty = true;
}

void VulkanCommandList::bindSamplerAt(uint32_t set, uint32_t binding,
                                      SamplerRef sampler) {
  if (set >= kVkCmdMaxDescriptorSets) {
    throw std::runtime_error("bindSamplerAt: set index >= kVkCmdMaxDescriptorSets");
  }
  if (!sampler) {
    throw std::runtime_error("bindSamplerAt: null SamplerRef");
  }
  PendingWrite w;
  w.binding = binding;
  w.type = VK_DESCRIPTOR_TYPE_SAMPLER;
  w.kind = PendingWrite::Kind::Image;
  w.imgInfo.sampler =
      static_cast<VulkanSampler*>(sampler.get())->vkHandle();
  w.imgInfo.imageView = VK_NULL_HANDLE;
  w.imgInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  m_pendingSets[set].writes.push_back(w);
  m_pendingSets[set].dirty = true;
}

void VulkanCommandList::pushAt(uint32_t offset, const void* data,
                               uint32_t size) {
  if (offset + size > kVkCmdPushConstantArenaBytes) {
    throw std::runtime_error("pushAt: offset+size exceeds 128 bytes");
  }
  std::memcpy(m_pendingPc.data() + offset, data, size);
  m_pcDirtyLo = std::min(m_pcDirtyLo, offset);
  m_pcDirtyHi = std::max(m_pcDirtyHi, offset + size);
}

void VulkanCommandList::bindResources(uint32_t set,
                                      std::span<const ResourceBinding> bindings) {
  for (const auto& rb : bindings) {
    std::visit(
        [&](const auto& res) {
          using T = std::decay_t<decltype(res)>;
          if constexpr (std::is_same_v<T, BufferRef>) {
            bindBufferAt(set, rb.binding, res);
          } else if constexpr (std::is_same_v<T, ImageBinding>) {
            bindImageAt(set, rb.binding, res.image, res.sampler);
          } else if constexpr (std::is_same_v<T, SamplerRef>) {
            bindSamplerAt(set, rb.binding, res);
          }
        },
        rb.resource);
  }
}

void VulkanCommandList::pushConstants(const PushConstants& pc) {
  auto v = pc.view();
  if (v.empty()) return;
  pushAt(0, v.data(), static_cast<uint32_t>(v.size()));
}

void VulkanCommandList::flushPending() {
  if (!m_boundPipeline) return;

  // ---- Descriptor sets ---------------------------------------------------
  for (uint32_t setIdx = 0; setIdx < m_pendingSets.size(); ++setIdx) {
    auto& s = m_pendingSets[setIdx];
    if (!s.dirty) continue;

    VkDescriptorSetLayout layout = m_boundPipeline->setLayout(setIdx);
    if (layout == VK_NULL_HANDLE) {
      s.clear();
      continue;
    }

    // Compute content hash for cache lookup in the allocator.
    uint64_t contentHash = computeDescriptorContentHash(m_boundPipeline, setIdx, s.writes);
    auto [ds, cacheHit] = m_device->descriptorAllocator().allocateOrReuse(layout, contentHash);

    if (cacheHit) {
      // Descriptor set already has the correct content — just bind it.
      vkCmdBindDescriptorSets(m_cmd, m_boundPipeline->bindPoint(), m_boundLayout,
                              setIdx, 1, &ds, 0, nullptr);
      s.clear();
      continue;
    }

    // Fresh allocation — write descriptors and bind.
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(s.writes.size());
    for (const auto& pw : s.writes) {
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = ds;
      w.dstBinding = pw.binding;
      w.dstArrayElement = 0;
      w.descriptorCount = 1;
      w.descriptorType = pw.type;
      if (pw.kind == PendingWrite::Kind::Buffer) {
        w.pBufferInfo = &pw.bufInfo;
      } else {
        w.pImageInfo = &pw.imgInfo;
      }
      writes.push_back(w);
    }

    if (!writes.empty()) {
      vkUpdateDescriptorSets(m_device->vkDevice(),
                             static_cast<uint32_t>(writes.size()), writes.data(),
                             0, nullptr);
    }

    vkCmdBindDescriptorSets(m_cmd, m_boundPipeline->bindPoint(), m_boundLayout,
                            setIdx, 1, &ds, 0, nullptr);
    s.clear();
  }

  // ---- Push constants ----------------------------------------------------
  if (m_pcDirtyHi > m_pcDirtyLo) {
    vkCmdPushConstants(m_cmd, m_boundLayout, m_boundPipeline->pushStageFlags(),
                       m_pcDirtyLo, m_pcDirtyHi - m_pcDirtyLo,
                       m_pendingPc.data() + m_pcDirtyLo);
    m_pcDirtyLo = kVkCmdPushConstantArenaBytes;
    m_pcDirtyHi = 0;
  }
}

void VulkanCommandList::rawDispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
  if (!m_boundPipeline) {
    throw std::runtime_error(
        "rawDispatch called without bindComputePipeline");
  }
  flushPending();
  vkCmdDispatch(m_cmd, gx, gy, gz);
}

// ---- R6: Graphics -------------------------------------------------------
void VulkanCommandList::beginRenderPass(const RenderPassBeginInfo& info) {
  std::vector<VkRenderingAttachmentInfo> colorAtts;
  colorAtts.reserve(info.colorAttachments.size());

  for (const auto& a : info.colorAttachments) {
    auto* vkImg = static_cast<VulkanImage*>(a.image.get());
    VkRenderingAttachmentInfo at{};
    at.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    at.imageView = vkImg->vkView();
    at.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    at.loadOp = (a.loadOp == RenderPassBeginInfo::Attachment::LoadOp::Clear)
                    ? VK_ATTACHMENT_LOAD_OP_CLEAR
                    : (a.loadOp == RenderPassBeginInfo::Attachment::LoadOp::Load
                           ? VK_ATTACHMENT_LOAD_OP_LOAD
                           : VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    at.storeOp = (a.storeOp == RenderPassBeginInfo::Attachment::StoreOp::Store)
                     ? VK_ATTACHMENT_STORE_OP_STORE
                     : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    std::memcpy(&at.clearValue.color, a.clearValue.colorF, sizeof(float) * 4);
    colorAtts.push_back(at);
  }

  VkRenderingAttachmentInfo depthAtt{};
  bool hasDepth = info.depthAttachment.has_value();
  if (hasDepth) {
    const auto& d = *info.depthAttachment;
    auto* vkImg = static_cast<VulkanImage*>(d.image.get());
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = vkImg->vkView();
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = (d.loadOp == RenderPassBeginInfo::Attachment::LoadOp::Clear)
                          ? VK_ATTACHMENT_LOAD_OP_CLEAR
                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {d.clearValue.depthStencil.depth,
                                        d.clearValue.depthStencil.stencil};
  }

  VkRenderingInfo ri{};
  ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  ri.renderArea = {{info.renderArea.x, info.renderArea.y},
                   {info.renderArea.width, info.renderArea.height}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = static_cast<uint32_t>(colorAtts.size());
  ri.pColorAttachments = colorAtts.data();
  ri.pDepthAttachment = hasDepth ? &depthAtt : nullptr;
  ri.pStencilAttachment = nullptr;

  vkCmdBeginRendering(m_cmd, &ri);
  m_inRenderPass = true;
}

void VulkanCommandList::endRenderPass() {
  vkCmdEndRendering(m_cmd);
  m_inRenderPass = false;
}

void VulkanCommandList::bindGraphicsPipeline(PipelineRef pipeline) {
  if (!pipeline) {
    throw std::runtime_error(
        "VulkanCommandList::bindGraphicsPipeline received null pipeline");
  }
  auto* vkPso = static_cast<VulkanPipeline*>(pipeline.get());
  if (vkPso->vkPipeline() == m_boundGraphicsPipelineHandle) {
    return;  // Same PSO already bound
  }
  vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPso->vkPipeline());

  m_boundGraphicsPipeline = vkPso;
  m_boundGraphicsPipelineHandle = vkPso->vkPipeline();
  m_boundGraphicsLayout = vkPso->vkPipelineLayout();

  // Also set m_boundPipeline so flushPending() works for graphics too
  m_boundPipeline = vkPso;
  m_boundPipelineHandle = vkPso->vkPipeline();
  m_boundLayout = vkPso->vkPipelineLayout();
  clearPending();

  // Clear vertex buffer tracking
  m_vertexBuffers.clear();
}

void VulkanCommandList::setViewport(const Viewport& vp) {
  VkViewport viewport{};
  viewport.x = vp.x;
  viewport.y = vp.y;
  viewport.width = vp.width;
  viewport.height = vp.height;
  viewport.minDepth = vp.minDepth;
  viewport.maxDepth = vp.maxDepth;
  vkCmdSetViewport(m_cmd, 0, 1, &viewport);
}

void VulkanCommandList::setScissor(const Rect2D& rect) {
  VkRect2D scissor{};
  scissor.offset = {rect.x, rect.y};
  scissor.extent = {rect.width, rect.height};
  vkCmdSetScissor(m_cmd, 0, 1, &scissor);
}

void VulkanCommandList::bindVertexBuffer(uint32_t binding, BufferRef buf,
                                         size_t offset) {
  if (!buf) {
    throw std::runtime_error("bindVertexBuffer: null BufferRef");
  }
  auto* vkBuf = static_cast<VulkanBuffer*>(buf.get());
  m_vertexBuffers[binding] = {vkBuf->vkHandle(),
                              static_cast<VkDeviceSize>(offset)};
}

void VulkanCommandList::bindIndexBuffer(BufferRef buf, IndexFormat fmt,
                                        size_t offset) {
  if (!buf) {
    throw std::runtime_error("bindIndexBuffer: null BufferRef");
  }
  auto* vkBuf = static_cast<VulkanBuffer*>(buf.get());
  m_indexBuffer = vkBuf->vkHandle();
  m_indexType = (fmt == IndexFormat::U16) ? VK_INDEX_TYPE_UINT16
                                          : VK_INDEX_TYPE_UINT32;
  m_indexBufferOffset = static_cast<VkDeviceSize>(offset);
  vkCmdBindIndexBuffer(m_cmd, m_indexBuffer, m_indexBufferOffset, m_indexType);
}

void VulkanCommandList::draw(uint32_t vertexCount, uint32_t instanceCount,
                             uint32_t firstVertex, uint32_t firstInstance) {
  if (!m_boundGraphicsPipeline) {
    throw std::runtime_error("draw called without bindGraphicsPipeline");
  }
  if (!m_inRenderPass) {
    throw std::runtime_error("draw called outside render pass");
  }

  // Bind accumulated vertex buffers
  if (!m_vertexBuffers.empty()) {
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceSize> offsets;
    for (const auto& [bindIdx, bufInfo] : m_vertexBuffers) {
      while (buffers.size() < bindIdx + 1) {
        buffers.push_back(VK_NULL_HANDLE);
        offsets.push_back(0);
      }
      buffers[bindIdx] = bufInfo.first;
      offsets[bindIdx] = bufInfo.second;
    }
    vkCmdBindVertexBuffers(m_cmd, 0, static_cast<uint32_t>(buffers.size()),
                           buffers.data(), offsets.data());
  }

  flushPending();
  vkCmdDraw(m_cmd, vertexCount, instanceCount, firstVertex, firstInstance);
}

void VulkanCommandList::drawIndexed(uint32_t indexCount, uint32_t instanceCount,
                                    uint32_t firstIndex, int32_t vertexOffset,
                                    uint32_t firstInstance) {
  if (!m_boundGraphicsPipeline) {
    throw std::runtime_error("drawIndexed called without bindGraphicsPipeline");
  }
  if (!m_inRenderPass) {
    throw std::runtime_error("drawIndexed called outside render pass");
  }

  // Bind accumulated vertex buffers
  if (!m_vertexBuffers.empty()) {
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceSize> offsets;
    for (const auto& [bindIdx, bufInfo] : m_vertexBuffers) {
      while (buffers.size() < bindIdx + 1) {
        buffers.push_back(VK_NULL_HANDLE);
        offsets.push_back(0);
      }
      buffers[bindIdx] = bufInfo.first;
      offsets[bindIdx] = bufInfo.second;
    }
    vkCmdBindVertexBuffers(m_cmd, 0, static_cast<uint32_t>(buffers.size()),
                           buffers.data(), offsets.data());
  }

  flushPending();
  vkCmdDrawIndexed(m_cmd, indexCount, instanceCount, firstIndex, vertexOffset,
                   firstInstance);
}

}  // namespace ksk::rhi::vulkan
