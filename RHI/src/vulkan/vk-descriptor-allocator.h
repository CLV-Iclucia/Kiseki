//
// vk-descriptor-allocator.h
// VkDescriptorSet allocator with content-hash caching and optional
// ring-of-pools for frame-pipelined rendering.
//
// Usage modes:
//
//   1. Single-pool (R0–R6, pure compute / tests):
//      Construct with just VkDevice. Use allocateOrReuse() / reset().
//      reset() recycles the entire pool — caller must have vkDeviceWaitIdle'd.
//
//   2. Ring-of-pools (R7+, frame-pipelined rendering):
//      After swapchain creation, call initRing(N) where N = imageCount.
//      Each frame:
//        - setActivePool(frameIdx) at frame start (after waitFence)
//        - resetActivePool() to recycle that frame's pool (safe: fence done)
//        - allocateOrReuse() goes to the active pool
//
//   Computing path (submitAndWait) continues to use the base pool (index 0)
//   and reset() after sync — unaffected by ring-of-pools.
//

#pragma once

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>

namespace sim::rhi::vulkan {

class DescriptorSetAllocator {
 public:
  explicit DescriptorSetAllocator(VkDevice device);
  ~DescriptorSetAllocator();

  struct Caps {
    uint32_t maxSets = 1024;
    uint32_t storageBuffers = 4096;
    uint32_t uniformBuffers = 512;
    uint32_t storageImages = 1024;
    uint32_t sampledImages = 1024;
    uint32_t samplers = 256;
  };

  // ---- Primary API ----------------------------------------------------------

  // Allocate (or reuse from cache) a descriptor set for the given layout.
  // `contentHash` identifies the bindings content: if a set with the same hash
  // was previously allocated in the current pool epoch, it is returned directly
  // without allocating or updating. Returns {set, true} on cache hit,
  // {set, false} on fresh allocation (caller must vkUpdateDescriptorSets).
  struct AllocResult {
    VkDescriptorSet set;
    bool cacheHit;
  };
  AllocResult allocateOrReuse(VkDescriptorSetLayout layout, uint64_t contentHash);

  // Legacy allocate without caching (always allocates fresh).
  VkDescriptorSet allocate(VkDescriptorSetLayout layout);

  // Reset ALL pools (single-pool mode) or just the base pool (ring mode).
  // Also clears the descriptor cache for affected pools.
  // Caller must ensure GPU is not referencing any allocated sets.
  void reset() noexcept;

  // ---- Ring-of-pools API (R7+) ---------------------------------------------

  // Initialize N additional pools for frame-pipelined usage.
  void initRing(uint32_t poolCount);

  // Set which pool subsequent allocate() calls draw from.
  void setActivePool(uint32_t index) noexcept;

  // Reset the currently active pool and its cache.
  void resetActivePool() noexcept;

  // Number of pools in the ring (1 if ring not initialized).
  uint32_t ringSize() const noexcept {
    return static_cast<uint32_t>(m_pools.size());
  }

 private:
  VkDescriptorPool createPool();

  VkDevice m_device;

  struct PoolSlot {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, VkDescriptorSet> cache;
  };

  std::vector<PoolSlot> m_pools;  // m_pools[0] is the "base" pool
  uint32_t m_activePoolIdx = 0;
};

}  // namespace sim::rhi::vulkan
