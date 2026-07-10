//
// buffer.h
// GPU buffer abstraction.
// See docs/rhi-plan.md §3.2.
//

#pragma once

#include <Core/properties.h>
#include <RHI/rc-ptr.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ksk::rhi {

struct BufferDesc {
  size_t sizeBytes = 0;

  enum class Visibility : uint32_t {
    DeviceLocal,   // GPU-only.   Vulkan: DEVICE_LOCAL,   DX12: DEFAULT  heap.
    HostVisible,   // CPU-write.  Vulkan: HOST_VISIBLE,   DX12: UPLOAD   heap.
    Readback,      // GPU-writes, CPU-reads.  DX12: READBACK heap.
  } visibility = Visibility::DeviceLocal;

  // Plain enum (not enum class) so we can OR-combine into a uint32_t mask.
  enum Usage : uint32_t {
    None = 0,
    Storage = 1u << 0,      // RW from compute / graphics
    Uniform = 1u << 1,      // Constant buffer
    Vertex = 1u << 2,
    Index = 1u << 3,
    Indirect = 1u << 4,
    TransferSrc = 1u << 5,
    TransferDst = 1u << 6,
  };
  uint32_t usage = 0;  // e.g. BufferDesc::Storage | BufferDesc::TransferDst

  std::string debugName;  // Used only when DeviceDesc::enableValidation is true.
};

// ---- Abstract Buffer -------------------------------------------------------
//
// Public, abstract; backend impls (`vulkan::VulkanBuffer`) derive and provide
// the actual handle. Each resource class hand-writes the 6-line refcount
// boilerplate below — there is intentionally no shared base class or CRTP
// (see plan §3.0 / R21).
//
class Buffer : public ksk::core::NonCopyable {
 public:
  // ---- Refcount (hand-written; six lines per resource class) -------------
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  // ---- Public interface --------------------------------------------------
  virtual ~Buffer() = default;
  virtual size_t sizeBytes() const = 0;
  virtual void* map() = 0;
  virtual void unmap() = 0;

  template <class T>
  std::span<T> mapTyped() {
    auto* p = static_cast<T*>(map());
    return p ? std::span<T>(p, sizeBytes() / sizeof(T)) : std::span<T>{};
  }

 protected:
  Buffer() = default;
  // Backend decides Tier 0 / Tier 1 destruction strategy.
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using BufferRef = RcPtr<Buffer>;

}  // namespace ksk::rhi
