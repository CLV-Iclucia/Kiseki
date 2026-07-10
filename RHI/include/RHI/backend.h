//
// backend.h
// Backend selection enums and DeviceDesc.
// See docs/rhi-plan.md §3.1.
//

#pragma once

#include <cstdint>

namespace ksk::rhi {

enum class Backend : uint32_t {
  Vulkan,
  Dx12,  // Reserved for R3+; not implemented in R0–R2.
};

enum class QueueType : uint32_t {
  Graphics,
  Compute,
  Transfer,
};

struct DeviceDesc {
  Backend backend = Backend::Vulkan;
  bool enableValidation = false;
  // Future: GPU picker, extension allowlist, feature opt-ins.
};

}  // namespace ksk::rhi
