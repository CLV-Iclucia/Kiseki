//
// vk-commands.h
//

#pragma once

#include <RHI/backend.h>
#include <RHI/commands.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace ksk::rhi::vulkan {

class VulkanDevice;
class VulkanPipeline;

// Vulkan 1.3 spec guarantees maxBoundDescriptorSets >= 4 — match
// kMaxDescriptorSets in vk-pipeline.h.
inline constexpr uint32_t kVkCmdMaxDescriptorSets = 4;
inline constexpr uint32_t kVkCmdPushConstantArenaBytes = 128;

class VulkanCommandList final : public CommandList {
 public:
  VulkanCommandList(VulkanDevice* device, QueueType queue, VkCommandBuffer cb);
  ~VulkanCommandList() override;

  // ---- R0–R2 surface ----------------------------------------------------
  void copyBuffer(BufferRef src, BufferRef dst,
                  std::span<const BufferCopy> regions) override;
  void copyBufferToImage(BufferRef src, ImageRef dst,
                         std::span<const BufferImageCopy> regions) override;
  void copyImageToBuffer(ImageRef src, BufferRef dst,
                         std::span<const BufferImageCopy> regions) override;
  void clearImage(ImageRef image, const ClearValue& value) override;
  void fillBuffer(BufferRef buffer, uint32_t value) override;

  void barrier(const BarrierDesc& desc) override;

  void debugMarkerBegin(std::string_view name) override;
  void debugMarkerEnd() override;

  void end() override;

  // ---- R4 / R5: compute -------------------------------------------------
  void bindComputePipeline(PipelineRef pipeline) override;

  void bindBufferAt(uint32_t set, uint32_t binding, BufferRef buf) override;
  void bindImageAt(uint32_t set, uint32_t binding, ImageRef img,
                   SamplerRef sampler) override;
  void bindSamplerAt(uint32_t set, uint32_t binding,
                     SamplerRef sampler) override;
  void pushAt(uint32_t offset, const void* data, uint32_t size) override;

  void bindResources(uint32_t set,
                     std::span<const ResourceBinding> bindings) override;
  void pushConstants(const PushConstants& pc) override;

  void rawDispatch(uint32_t gx, uint32_t gy, uint32_t gz) override;

  // ---- R6: Graphics -------------------------------------------------
  void beginRenderPass(const RenderPassBeginInfo&) override;
  void endRenderPass() override;

  void bindGraphicsPipeline(PipelineRef) override;

  void setViewport(const Viewport&) override;
  void setScissor(const Rect2D&) override;

  void bindVertexBuffer(uint32_t binding, BufferRef, size_t offset = 0) override;
  void bindIndexBuffer(BufferRef, IndexFormat, size_t offset = 0) override;

  void draw(uint32_t vertexCount, uint32_t instanceCount = 1,
            uint32_t firstVertex = 0, uint32_t firstInstance = 0) override;
  void drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                   uint32_t firstIndex = 0, int32_t vertexOffset = 0,
                   uint32_t firstInstance = 0) override;

  // ---- Backend-internal -------------------------------------------------
  VkCommandBuffer vkCommandBuffer() const noexcept { return m_cmd; }
  QueueType queueType() const noexcept { return m_queue; }

  // Called by VulkanDevice::submit to ensure recording is closed exactly once.
  void endIfRecording();

 private:
  // Pending descriptor writes per set. A "write" stores its own backing
  // VkDescriptor*Info inline so we never juggle dangling pointers across
  // vector reallocations — VkWriteDescriptorSet entries are synthesised at
  // flush time from these records, with pBufferInfo / pImageInfo pointing
  // into the same `PendingWrite` slot.
  struct PendingWrite {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    enum class Kind : uint8_t { Buffer, Image } kind = Kind::Buffer;
    VkDescriptorBufferInfo bufInfo{};
    VkDescriptorImageInfo imgInfo{};
  };

  struct PendingSet {
    bool dirty = false;
    std::vector<PendingWrite> writes;
    void clear() {
      dirty = false;
      writes.clear();
    }
  };

  void clearPending();
  void flushPending();
  VkDescriptorType lookupDescriptorType(uint32_t set, uint32_t binding) const;
  static uint64_t computeDescriptorContentHash(const VulkanPipeline* pipeline,
                                               uint32_t setIdx,
                                               const std::vector<PendingWrite>& writes);

  VulkanDevice* m_device;
  QueueType m_queue;
  VkCommandBuffer m_cmd;
  bool m_recording = true;
  int m_markerDepth = 0;

  // Currently bound compute pipeline state.
  VulkanPipeline* m_boundPipeline = nullptr;
  VkPipeline m_boundPipelineHandle = VK_NULL_HANDLE;
  VkPipelineLayout m_boundLayout = VK_NULL_HANDLE;

  std::array<PendingSet, kVkCmdMaxDescriptorSets> m_pendingSets{};

  // Push constants arena. lo/hi track the dirty range; hi == lo means clean.
  std::array<std::byte, kVkCmdPushConstantArenaBytes> m_pendingPc{};
  uint32_t m_pcDirtyLo = kVkCmdPushConstantArenaBytes;
  uint32_t m_pcDirtyHi = 0;

  // R6: Graphics rendering state
  bool m_inRenderPass = false;
  VulkanPipeline* m_boundGraphicsPipeline = nullptr;
  VkPipeline m_boundGraphicsPipelineHandle = VK_NULL_HANDLE;
  VkPipelineLayout m_boundGraphicsLayout = VK_NULL_HANDLE;
  std::map<uint32_t, std::pair<VkBuffer, VkDeviceSize>> m_vertexBuffers{};
  VkBuffer m_indexBuffer = VK_NULL_HANDLE;
  VkIndexType m_indexType = VK_INDEX_TYPE_UINT32;
  VkDeviceSize m_indexBufferOffset = 0;
};


}  // namespace ksk::rhi::vulkan
