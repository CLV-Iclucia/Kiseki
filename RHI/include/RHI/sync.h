//
// sync.h
// Sync primitives: Fence (host-visible) and Semaphore (queue-to-queue).
// Single-ownership, NOT ref-counted (see docs/rhi-plan.md §3.7 / R17).
//

#pragma once

#include <Core/properties.h>

namespace ksk::rhi {

class Fence : public ksk::core::NonCopyable {
 public:
  virtual ~Fence() = default;

 protected:
  Fence() = default;
};

class Semaphore : public ksk::core::NonCopyable {
 public:
  virtual ~Semaphore() = default;

 protected:
  Semaphore() = default;
};

}  // namespace ksk::rhi
