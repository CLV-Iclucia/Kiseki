//
// vk-device.cc
//
// Vulkan device factory + main lifecycle. Tier 0 sync model only (R0–R2);
// Tier 1 deferred destroy is in R8.
//

#define VMA_IMPLEMENTATION
#include "vk-device.h"
#include <RHI/shader-compiler.h>

#include "vk-buffer.h"
#include "vk-commands.h"
#include "vk-descriptor-allocator.h"
#include "vk-image.h"
#include "vk-internals.h"
#include "vk-pipeline-cache.h"
#include "vk-pipeline.h"
#include "vk-shader.h"
#include "vk-swapchain.h"
#include "vk-sync.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

namespace sim::rhi {

// ---- Device::create factory ------------------------------------------------
std::unique_ptr<Device> Device::create(const DeviceDesc& desc) {
  if (desc.backend != Backend::Vulkan) {
    spdlog::error("Rhi: only Vulkan backend is supported in R0–R2");
    return nullptr;
  }
  try {
    return std::make_unique<vulkan::VulkanDevice>(desc);
  } catch (const std::exception& e) {
    spdlog::error("Rhi: failed to create Vulkan device: {}", e.what());
    return nullptr;
  }
}

// ---- Device::createShaderCompiler -------------------------------------------
std::unique_ptr<ShaderCompiler> Device::createShaderCompiler() const {
  return ShaderCompiler::create(backend());
}

}  // namespace sim::rhi

namespace sim::rhi::vulkan {

// ---- ctor / dtor -----------------------------------------------------------
VulkanDevice::VulkanDevice(const DeviceDesc& desc) {
  initInstance(desc.enableValidation);
  initPhysicalDeviceAndDevice();
  initVma();
  initCommandPools();
  initDescriptorAllocator();
  m_pipelineCache = std::make_unique<VulkanPipelineCache>(m_device);
  spdlog::info("Rhi: Vulkan device ready (validation={})", m_validationEnabled);
}

VulkanDevice::~VulkanDevice() {
  if (m_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device);
  }

  // Reverse-order teardown per plan §6.2.
  // Pipeline cache must be destroyed before descriptor pool (pipelines may hold
  // descriptor set layouts, so drop refs first).
  m_pipelineCache.reset();

  // Descriptor allocator owns a VkDescriptorPool -- destroy before VkDevice.
  m_descriptorAlloc.reset();

  for (auto& qp : m_queuePools) {
    if (qp.pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(m_device, qp.pool, nullptr);
      qp.pool = VK_NULL_HANDLE;
    }
  }

  if (m_allocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(m_allocator);
    m_allocator = VK_NULL_HANDLE;
  }

  if (m_device != VK_NULL_HANDLE) {
    vkb::destroy_device(m_vkbDevice);
    m_device = VK_NULL_HANDLE;
  }

  if (m_instance != VK_NULL_HANDLE) {
    vkb::destroy_instance(m_vkbInstance);
    m_instance = VK_NULL_HANDLE;
  }
}

// ---- Init: Instance --------------------------------------------------------
void VulkanDevice::initInstance(bool enableValidation) {
  vkb::InstanceBuilder ib;
  ib.set_app_name("SimCraft")
    .set_engine_name("SimCraft RHI")
    .require_api_version(1, 3, 0);
  if (enableValidation) {
    ib.request_validation_layers(true)
      .use_default_debug_messenger();
  }

  auto inst_ret = ib.build();
  if (!inst_ret) {
    throw std::runtime_error(
        std::string("vkb::InstanceBuilder::build failed: ") +
        inst_ret.error().message());
  }
  m_vkbInstance = inst_ret.value();
  m_instance = m_vkbInstance.instance;
  m_validationEnabled =
      enableValidation && m_vkbInstance.debug_messenger != VK_NULL_HANDLE;
}

// ---- Init: PhysicalDevice + Device ----------------------------------------
void VulkanDevice::initPhysicalDeviceAndDevice() {
  // Vulkan 1.3 features needed for sync2 + dynamic_rendering.
  VkPhysicalDeviceVulkan13Features feats13{};
  feats13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  feats13.synchronization2 = VK_TRUE;
  feats13.dynamicRendering = VK_TRUE;

  VkPhysicalDeviceVulkan12Features feats12{};
  feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  feats12.bufferDeviceAddress = VK_TRUE;
  feats12.timelineSemaphore = VK_TRUE;

  vkb::PhysicalDeviceSelector pds(m_vkbInstance);
  // No surface required for headless RHI usage in R0–R2.
  pds.defer_surface_initialization()
     .set_minimum_version(1, 3)
     .set_required_features_13(feats13)
     .set_required_features_12(feats12);

  auto phys_ret = pds.select();
  if (!phys_ret) {
    throw std::runtime_error(
        std::string("vkb::PhysicalDeviceSelector::select failed: ") +
        phys_ret.error().message());
  }
  vkb::PhysicalDevice phys = phys_ret.value();
  m_physicalDevice = phys.physical_device;

  vkb::DeviceBuilder db(phys);
  auto dev_ret = db.build();
  if (!dev_ret) {
    throw std::runtime_error(
        std::string("vkb::DeviceBuilder::build failed: ") +
        dev_ret.error().message());
  }
  m_vkbDevice = dev_ret.value();
  m_device = m_vkbDevice.device;

  // Resolve queues. Compute / Transfer fall back to Graphics if dedicated
  // families are unavailable, matching plan §6.3.
  auto pickQueue = [&](vkb::QueueType type) -> std::pair<VkQueue, uint32_t> {
    auto q = m_vkbDevice.get_queue(type);
    auto fi = m_vkbDevice.get_queue_index(type);
    if (q && fi) return {q.value(), fi.value()};
    auto qg = m_vkbDevice.get_queue(vkb::QueueType::graphics);
    auto fig = m_vkbDevice.get_queue_index(vkb::QueueType::graphics);
    return {qg ? qg.value() : VK_NULL_HANDLE, fig ? fig.value() : ~0u};
  };

  auto [gq, gqf] = pickQueue(vkb::QueueType::graphics);
  auto [cq, cqf] = pickQueue(vkb::QueueType::compute);
  auto [tq, tqf] = pickQueue(vkb::QueueType::transfer);

  m_queuePools[(int)QueueType::Graphics] = {gqf, gq, VK_NULL_HANDLE};
  m_queuePools[(int)QueueType::Compute]  = {cqf, cq, VK_NULL_HANDLE};
  m_queuePools[(int)QueueType::Transfer] = {tqf, tq, VK_NULL_HANDLE};

  spdlog::info("Rhi: queues — graphics(family={}) compute(family={}) transfer(family={})",
               gqf, cqf, tqf);
}

// ---- Init: VMA -------------------------------------------------------------
void VulkanDevice::initVma() {
  VmaAllocatorCreateInfo info{};
  info.physicalDevice = m_physicalDevice;
  info.device = m_device;
  info.instance = m_instance;
  info.vulkanApiVersion = VK_API_VERSION_1_3;
  info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  VK_CHECK(vmaCreateAllocator(&info, &m_allocator));
}

// ---- Init: command pools ---------------------------------------------------
void VulkanDevice::initCommandPools() {
  for (auto& qp : m_queuePools) {
    if (qp.familyIndex == ~0u || qp.queue == VK_NULL_HANDLE) continue;
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
               VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    ci.queueFamilyIndex = qp.familyIndex;
    VK_CHECK(vkCreateCommandPool(m_device, &ci, nullptr, &qp.pool));
  }
}

// ---- Init: descriptor pool ------------------------------------------------
void VulkanDevice::initDescriptorAllocator() {
  m_descriptorAlloc = std::make_unique<DescriptorSetAllocator>(m_device);
}

// ---- Resource creation -----------------------------------------------------
// NOTE: use RcPtr<T>(T*) ctor (which calls addRef → m_rc=1), NOT adopt().
// adopt() is reserved for the rare case where the caller already addRef'd
// the raw pointer; misusing it leaks the resource (m_rc stays at 0, so
// the very first release underflows and never calls destroy()).
BufferRef VulkanDevice::createBuffer(const BufferDesc& desc) {
  return BufferRef(new VulkanBuffer(this, desc));
}

ImageRef VulkanDevice::createImage(const ImageDesc& desc) {
  return ImageRef(new VulkanImage(this, desc));
}

SamplerRef VulkanDevice::createSampler(const SamplerDesc& desc) {
  return SamplerRef(new VulkanSampler(this, desc));
}

// ---- R4: shaders / pipelines ----------------------------------------------
ShaderRef VulkanDevice::createShader(std::span<const std::byte> bytecode,
                                     ShaderStage stage,
                                     std::string_view entryPoint) {
  return ShaderRef(new VulkanShader(this, bytecode, stage, entryPoint));
}

PipelineRef VulkanDevice::createComputePipeline(
    const ComputePipelineDesc& desc) {
  if (auto cached = m_pipelineCache->findCompute(desc)) return cached;
  auto pipeline = PipelineRef(new VulkanPipeline(this, desc));
  m_pipelineCache->insertCompute(desc, pipeline);
  return pipeline;
}

PipelineRef VulkanDevice::createGraphicsPipeline(
    const GraphicsPipelineDesc& desc) {
  if (auto cached = m_pipelineCache->findGraphics(desc)) return cached;
  auto pipeline = PipelineRef(new VulkanPipeline(this, desc));
  m_pipelineCache->insertGraphics(desc, pipeline);
  return pipeline;
}

// ---- R7: Swapchain ---------------------------------------------------------
std::unique_ptr<Swapchain> VulkanDevice::createSwapchain(
    const SwapchainDesc& desc) {
  try {
    return std::make_unique<VulkanSwapchain>(this, desc);
  } catch (const std::exception& e) {
    spdlog::error("Rhi: failed to create swapchain: {}", e.what());
    return nullptr;
  }
}

// ---- Sync ------------------------------------------------------------------
std::unique_ptr<Fence> VulkanDevice::createFence() {
  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  // Created unsignaled.
  VkFence handle = VK_NULL_HANDLE;
  VK_CHECK(vkCreateFence(m_device, &info, nullptr, &handle));
  return std::make_unique<VulkanFence>(this, handle);
}

std::unique_ptr<Semaphore> VulkanDevice::createSemaphore() {
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore handle = VK_NULL_HANDLE;
  VK_CHECK(vkCreateSemaphore(m_device, &info, nullptr, &handle));
  return std::make_unique<VulkanSemaphore>(this, handle);
}

void VulkanDevice::waitFence(Fence& fence) {
  auto* f = static_cast<VulkanFence*>(&fence);
  VkFence h = f->vkHandle();
  VK_CHECK(vkWaitForFences(m_device, 1, &h, VK_TRUE, UINT64_MAX));
}

bool VulkanDevice::isFenceSignaled(Fence& fence) {
  auto* f = static_cast<VulkanFence*>(&fence);
  VkResult r = vkGetFenceStatus(m_device, f->vkHandle());
  return r == VK_SUCCESS;
}

void VulkanDevice::resetFence(Fence& fence) {
  auto* f = static_cast<VulkanFence*>(&fence);
  VkFence h = f->vkHandle();
  VK_CHECK(vkResetFences(m_device, 1, &h));
}

// ---- Command list creation -------------------------------------------------
std::unique_ptr<CommandList> VulkanDevice::beginCommands(QueueType queue) {
  const auto& qp = m_queuePools[(int)queue];
  if (qp.pool == VK_NULL_HANDLE) {
    throw std::runtime_error("Rhi: requested queue type has no command pool");
  }

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = qp.pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &cb));

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cb, &bi));

  return std::make_unique<VulkanCommandList>(this, queue, cb);
}

// ---- Submit ----------------------------------------------------------------
void VulkanDevice::submit(CommandList& cmd,
                          std::span<Semaphore* const> waitSemaphores,
                          std::span<Semaphore* const> signalSemaphores,
                          Fence* onComplete) {
  auto* vcmd = static_cast<VulkanCommandList*>(&cmd);
  vcmd->endIfRecording();

  std::vector<VkSemaphoreSubmitInfo> waitInfos;
  waitInfos.reserve(waitSemaphores.size());
  for (Semaphore* s : waitSemaphores) {
    if (!s) continue;
    VkSemaphoreSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    si.semaphore = static_cast<VulkanSemaphore*>(s)->vkHandle();
    si.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    waitInfos.push_back(si);
  }

  std::vector<VkSemaphoreSubmitInfo> signalInfos;
  signalInfos.reserve(signalSemaphores.size());
  for (Semaphore* s : signalSemaphores) {
    if (!s) continue;
    VkSemaphoreSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    si.semaphore = static_cast<VulkanSemaphore*>(s)->vkHandle();
    si.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signalInfos.push_back(si);
  }

  VkCommandBufferSubmitInfo cbi{};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  cbi.commandBuffer = vcmd->vkCommandBuffer();

  VkSubmitInfo2 si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  si.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
  si.pWaitSemaphoreInfos = waitInfos.data();
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbi;
  si.signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size());
  si.pSignalSemaphoreInfos = signalInfos.data();

  VkFence fence = onComplete
                      ? static_cast<VulkanFence*>(onComplete)->vkHandle()
                      : VK_NULL_HANDLE;
  VkQueue q = m_queuePools[(int)vcmd->queueType()].queue;
  VK_CHECK(vkQueueSubmit2(q, 1, &si, fence));
}

void VulkanDevice::submitAndWait(CommandList& cmd, QueueType queue) {
  submit(cmd, {}, {}, nullptr);
  waitQueueIdle(queue);
}

// ---- Internal accessors ----------------------------------------------------
VkQueue VulkanDevice::queueForType(QueueType q) const noexcept {
  return m_queuePools[(int)q].queue;
}
uint32_t VulkanDevice::queueFamilyForType(QueueType q) const noexcept {
  return m_queuePools[(int)q].familyIndex;
}

void VulkanDevice::waitQueueIdle(QueueType q) noexcept {
  VkQueue queue = m_queuePools[(int)q].queue;
  if (queue != VK_NULL_HANDLE) {
    vkQueueWaitIdle(queue);
  }
}

VkQueue VulkanDevice::resolveQueueForCommandList(CommandList& cmd) const {
  auto* vcmd = static_cast<VulkanCommandList*>(&cmd);
  return m_queuePools[(int)vcmd->queueType()].queue;
}

}  // namespace sim::rhi::vulkan
