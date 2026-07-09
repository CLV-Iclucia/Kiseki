//
// vk-device.h
// Vulkan implementation of Device.
// See docs/rhi-plan.md §3.1 / §6.2.
//

#pragma once

#include <RHI/device.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <VkBootstrap.h>

#include <array>
#include <memory>

#include "vk-pipeline-cache.h"

namespace sim::rhi::vulkan {

class DescriptorSetAllocator;

// Per-queue transient command pool (used by beginCommands).
struct QueuePool {
  uint32_t familyIndex = ~0u;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
};

class VulkanDevice final : public Device {
 public:
  // Construction always succeeds or throws std::runtime_error.
  // The factory `Device::create()` catches and converts to nullptr.
  VulkanDevice(const DeviceDesc& desc);
  ~VulkanDevice() override;

  // ---- Device interface -------------------------------------------------
  BufferRef createBuffer(const BufferDesc& desc) override;
  ImageRef createImage(const ImageDesc& desc) override;
  SamplerRef createSampler(const SamplerDesc& desc) override;

  ShaderRef createShader(std::span<const std::byte> bytecode,
                         ShaderStage stage,
                         std::string_view entryPoint) override;
  PipelineRef createComputePipeline(const ComputePipelineDesc& desc) override;
  PipelineRef createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;

  std::unique_ptr<Swapchain> createSwapchain(const SwapchainDesc& desc) override;

  std::unique_ptr<Fence> createFence() override;
  std::unique_ptr<Semaphore> createSemaphore() override;
  void waitFence(Fence& fence) override;
  bool isFenceSignaled(Fence& fence) override;
  void resetFence(Fence& fence) override;

  std::unique_ptr<CommandList> beginCommands(QueueType queue) override;

  void submit(CommandList& cmd,
              std::span<Semaphore* const> waitSemaphores,
              std::span<Semaphore* const> signalSemaphores,
              Fence* onComplete) override;

  void submitAndWait(CommandList& cmd, QueueType queue) override;

  [[nodiscard]] Backend backend() const override { return Backend::Vulkan; }
  bool frameLoopActive() const override { return m_frameLoopActive; }

  // ---- Backend-internal accessors --------------------------------------
  VkInstance vkInstance() const noexcept { return m_instance; }
  VkPhysicalDevice vkPhysicalDevice() const noexcept { return m_physicalDevice; }
  VkDevice vkDevice() const noexcept { return m_device; }
  VmaAllocator vmaAllocator() const noexcept { return m_allocator; }

  VkQueue queueForType(QueueType q) const noexcept;
  uint32_t queueFamilyForType(QueueType q) const noexcept;

  // Block until the chosen queue is idle (used by Tier 0 destroy paths).
  void waitQueueIdle(QueueType q) noexcept;

  // True once enableValidation was honored.
  bool validationEnabled() const noexcept { return m_validationEnabled; }

  // Descriptor set allocator owned by Device. Single instance shared across
  // command lists; safe under R0 single-thread record assumption.
  DescriptorSetAllocator& descriptorAllocator() noexcept {
    return *m_descriptorAlloc;
  }

 private:
  struct TypedShaderCache;

  void initInstance(bool enableValidation);
  void initPhysicalDeviceAndDevice();
  void initVma();
  void initCommandPools();
  void initDescriptorAllocator();

  // Detail used by submit().
  VkQueue resolveQueueForCommandList(CommandList& cmd) const;

  PipelineRef resolveTypedComputePipeline(
      const std::filesystem::path& source,
      const ShaderCompileOptions& options) override;

  // ---- vk-bootstrap state (kept around for clean teardown) -------------
  vkb::Instance m_vkbInstance{};
  vkb::Device m_vkbDevice{};

  // ---- Raw handles -----------------------------------------------------
  VkInstance m_instance = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;

  std::array<QueuePool, 3> m_queuePools{};  // index by QueueType

  std::unique_ptr<DescriptorSetAllocator> m_descriptorAlloc;
  std::unique_ptr<VulkanPipelineCache> m_pipelineCache;
  std::unique_ptr<TypedShaderCache> m_typedShaderCache;

  bool m_validationEnabled = false;
  bool m_frameLoopActive = false;  // R0–R2: never set
};

}  // namespace sim::rhi::vulkan
