//
// image.h
// GPU image, sampler, and image-binding abstractions.
// See docs/rhi-plan.md §3.3.
//

#pragma once

#include <Core/properties.h>
#include <RHI/format.h>
#include <RHI/rc-ptr.h>

#include <atomic>
#include <cstdint>
#include <string>

namespace ksk::rhi {

struct ImageDesc {
  enum class Dim : uint32_t { D1, D2, D3 };

  Dim dim = Dim::D3;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 1;
  Format format = Format::Undefined;
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;

  enum Usage : uint32_t {
    None = 0,
    Storage = 1u << 0,
    Sampled = 1u << 1,
    ColorAttachment = 1u << 2,
    DepthAttachment = 1u << 3,
    TransferSrc = 1u << 4,
    TransferDst = 1u << 5,
  };
  uint32_t usage = 0;

  std::string debugName;
};

// ---- Abstract Image --------------------------------------------------------
class Image : public ksk::core::NonCopyable {
 public:
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  virtual ~Image() = default;
  virtual ImageDesc desc() const = 0;

 protected:
  Image() = default;
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using ImageRef = RcPtr<Image>;

// ---- Sampler ---------------------------------------------------------------
struct SamplerDesc {
  enum class Filter : uint32_t { Nearest, Linear };
  enum class AddressMode : uint32_t { Repeat, ClampToEdge, ClampToBorder };

  Filter magFilter = Filter::Linear;
  Filter minFilter = Filter::Linear;
  AddressMode addressMode = AddressMode::ClampToEdge;
};

class Sampler : public ksk::core::NonCopyable {
 public:
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  virtual ~Sampler() = default;

 protected:
  Sampler() = default;
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using SamplerRef = RcPtr<Sampler>;

// ---- ImageBinding ----------------------------------------------------------
//
// Combines an Image with an optional Sampler, expressing both pure-storage
// usage (sampler empty) and combined sampled usage.
//
struct ImageBinding {
  ImageRef image;
  SamplerRef sampler;  // empty => pure storage / load-store
};

}  // namespace ksk::rhi
