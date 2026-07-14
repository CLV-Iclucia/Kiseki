//
// commands.h
// Command list — full surface (R0–R5).
// See docs/rhi-plan.md §3.5 / §3.6.
//
// Three compute tiers (plan §3.5.1):
//   • High level   — `dispatch(pso, params, gx, gy, gz)` template, recommended
//                    path. Internally: bindComputePipeline → resolve(first
//                    time only) → apply → rawDispatch.
//   • Mid level    — `bindBufferAt / bindImageAt / bindSamplerAt / pushAt`,
//                    used by SHADER_PARAMS::_apply and exposed for
//                    one-off-binding cases.
//   • Low fallback — `bindComputePipeline / bindResources / pushConstants /
//                    rawDispatch`, original "register the whole set in one
//                    shot" pattern.
//
// On Vulkan, descriptor writes don't get sent to the GPU until the implicit
// `flush_pending_` runs at `rawDispatch` (or `rawDraw`). See plan §13.3 for
// the state-machine in `vk-commands.cc`.
//

#pragma once

#include <Core/properties.h>
#include <RHI/buffer.h>
#include <RHI/image.h>
#include <RHI/pipeline.h>
#include <RHI/shader-params.h>

#include <array>
#include <cstddef>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace ksk::rhi {

template <class Derived>
class ComputeShader;

// ---- Buffer / image copies -------------------------------------------------
struct BufferCopy {
  size_t srcOffset = 0;
  size_t dstOffset = 0;
  size_t size = 0;
};

struct BufferImageCopy {
  size_t bufferOffset = 0;
  uint32_t bufferRowLength = 0;    // 0 = tightly packed (matches image extent)
  uint32_t bufferImageHeight = 0;  // 0 = tightly packed
  uint32_t mipLevel = 0;
  uint32_t baseArrayLayer = 0;
  uint32_t layerCount = 1;
  uint32_t imageOffsetX = 0;
  uint32_t imageOffsetY = 0;
  uint32_t imageOffsetZ = 0;
  uint32_t imageExtentW = 0;
  uint32_t imageExtentH = 0;
  uint32_t imageExtentD = 1;
};

// ---- Clear value -----------------------------------------------------------
struct ClearValue {
  union {
    float colorF[4];
    uint32_t colorU[4];
    int32_t colorI[4];
    struct {
      float depth;
      uint32_t stencil;
    } depthStencil;
  };

  static ClearValue makeColorF(float r, float g, float b, float a) {
    ClearValue v{};
    v.colorF[0] = r;
    v.colorF[1] = g;
    v.colorF[2] = b;
    v.colorF[3] = a;
    return v;
  }
  static ClearValue makeColorU(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    ClearValue v{};
    v.colorU[0] = r;
    v.colorU[1] = g;
    v.colorU[2] = b;
    v.colorU[3] = a;
    return v;
  }
  static ClearValue makeDepth(float d, uint32_t s = 0) {
    ClearValue v{};
    v.depthStencil.depth = d;
    v.depthStencil.stencil = s;
    return v;
  }
};

// ---- Barrier ---------------------------------------------------------------
struct BarrierDesc {
  enum Stage : uint32_t {
    StageNone = 0,
    StageComputeShader = 1u << 0,
    StageVertexShader = 1u << 1,
    StageFragmentShader = 1u << 2,
    StageTransfer = 1u << 3,
    StageAllGraphics = 1u << 4,
    StageTopOfPipe = 1u << 5,
    StageBottomOfPipe = 1u << 6,
    StageColorAttachmentOutput = 1u << 7,
    StageEarlyFragmentTests = 1u << 8,
    StageLateFragmentTests = 1u << 9,
  };
  enum Access : uint32_t {
    AccessNone = 0,
    AccessShaderRead = 1u << 0,
    AccessShaderWrite = 1u << 1,
    AccessTransferRead = 1u << 2,
    AccessTransferWrite = 1u << 3,
    AccessVertexRead = 1u << 4,
    AccessIndexRead = 1u << 5,
    AccessUniformRead = 1u << 6,
    AccessColorAttachmentRead = 1u << 7,
    AccessColorAttachmentWrite = 1u << 8,
    AccessDepthStencilRead = 1u << 9,
    AccessDepthStencilWrite = 1u << 10,
  };

  uint32_t srcStage = 0;
  uint32_t dstStage = 0;
  uint32_t srcAccess = 0;
  uint32_t dstAccess = 0;

  struct ImageBarrier {
    ImageRef image;
    enum class Layout : uint32_t {
      Undefined,
      General,
      ShaderReadOnly,
      TransferSrc,
      TransferDst,
      ColorAttachment,
      DepthAttachment,
      Present,
    };
    Layout oldLayout = Layout::Undefined;
    Layout newLayout = Layout::Undefined;
  };
  std::vector<ImageBarrier> imageBarriers;
};

// ---- Graphics render pass & state (R6) ------------------------------------
struct Viewport {
  float x = 0, y = 0;
  float width, height;
  float minDepth = 0.0f, maxDepth = 1.0f;
};

struct Rect2D {
  int32_t x = 0, y = 0;
  uint32_t width, height;
};

enum class IndexFormat { U16, U32 };

struct RenderPassBeginInfo {
  struct Attachment {
    ImageRef image;
    enum class LoadOp { Load, Clear, DontCare } loadOp = LoadOp::Load;
    enum class StoreOp { Store, DontCare } storeOp = StoreOp::Store;
    ClearValue clearValue{};  // Reuse existing ClearValue
  };
  using Attachments = std::vector<Attachment>;

  Attachments colorAttachments;
  std::optional<Attachment> depthAttachment;
  Rect2D renderArea;
};

// ---- Resource binding (low-tier) -------------------------------------------
//
// One slot of a descriptor set when using the "shove the whole set in" path.
// Variant keeps a single user-facing type for buffer / image+sampler /
// standalone sampler cases.
struct ResourceBinding {
  uint32_t binding = 0;
  std::variant<BufferRef, ImageBinding, SamplerRef> resource;
};

// ---- Push constants (low-tier helper) --------------------------------------
//
// 128-byte arena (Vulkan 1.0 minimum maxPushConstantsSize). type-safe
// `append<T>` accumulates trivially-copyable values; consumed via `pushAt`.
class PushConstants {
 public:
  template <typename T>
  void append(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (m_size + sizeof(T) > m_bytes.size()) {
      throw std::runtime_error("PushConstants overflow (>128B)");
    }
    std::memcpy(m_bytes.data() + m_size, &v, sizeof(T));
    m_size += static_cast<uint32_t>(sizeof(T));
  }
  std::span<const std::byte> view() const noexcept {
    return {m_bytes.data(), m_size};
  }
  void clear() noexcept { m_size = 0; }

 private:
  std::array<std::byte, 128> m_bytes{};
  uint32_t m_size = 0;
};

// ---- CommandList -----------------------------------------------------------
class CommandList : public ksk::core::NonCopyable {
 public:
  virtual ~CommandList() = default;

  // ============ Transfer (R0–R2) ============
  virtual void copyBuffer(BufferRef src, BufferRef dst,
                          std::span<const BufferCopy> regions) = 0;
  virtual void copyBufferToImage(BufferRef src, ImageRef dst,
                                 std::span<const BufferImageCopy> regions) = 0;
  virtual void copyImageToBuffer(ImageRef src, BufferRef dst,
                                 std::span<const BufferImageCopy> regions) = 0;
  virtual void clearImage(ImageRef image, const ClearValue& value) = 0;
  virtual void fillBuffer(BufferRef buffer, uint32_t value) = 0;

  // ============ Sync (R0–R2) ============
  virtual void barrier(const BarrierDesc& desc) = 0;

  // Convenience: global memory barrier without image layout transitions.
  // Default access flags cover the most common case (shader write → shader
  // read+write), override via the last two parameters when needed.
  void memoryBarrier(uint32_t srcStage, uint32_t dstStage,
                     uint32_t srcAccess = BarrierDesc::AccessShaderWrite,
                     uint32_t dstAccess = BarrierDesc::AccessShaderRead |
                                          BarrierDesc::AccessShaderWrite) {
    barrier({.srcStage = srcStage, .dstStage = dstStage,
             .srcAccess = srcAccess, .dstAccess = dstAccess});
  }

  // ============ Debug markers ============
  virtual void debugMarkerBegin(std::string_view name) = 0;
  virtual void debugMarkerEnd() = 0;

  // ============ Compute — low tier (R4) ============
  // Switch the bound pipeline. Clears any pending binds (set layouts may
  // change) — see plan §13.3.2.
  virtual void bindComputePipeline(PipelineRef pipeline) = 0;

  // ============ Compute — mid tier (R5; used by SHADER_PARAMS) ============
  virtual void bindBufferAt(uint32_t set, uint32_t binding, BufferRef buf) = 0;
  virtual void bindImageAt(uint32_t set, uint32_t binding, ImageRef img,
                           SamplerRef sampler = {}) = 0;
  virtual void bindSamplerAt(uint32_t set, uint32_t binding,
                             SamplerRef sampler) = 0;
  virtual void pushAt(uint32_t offset, const void* data, uint32_t size) = 0;

  // ============ Compute — low-tier fallback bulk binds (R5) ============
  virtual void bindResources(uint32_t set,
                             std::span<const ResourceBinding> bindings) = 0;
  virtual void pushConstants(const PushConstants& pc) = 0;

  // ============ Compute — dispatch (R5) ============
  virtual void rawDispatch(uint32_t gx, uint32_t gy, uint32_t gz) = 0;

  // High-tier three-in-one dispatch. Templated so the user's SHADER_PARAMS
  // type Self is visible at the call site; internally we delegate to the
  // non-template `ShaderParamsBase::_resolve` / `_apply`, so this is a thin
  // wrapper.
  //
  // Sentinel-driven re-resolution: if the same `params` instance is fed to a
  // different pipeline, we silently re-resolve. Steady-state cost is ONE
  // pointer compare in the cold branch, predictable jump table on the
  // applyParams switch.
  template <class P>
    requires std::is_base_of_v<ShaderParamsBase, P>
  void dispatch(PipelineRef pso, P& params,
                uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    bindComputePipeline(pso);
    const void* sig = pso.get();  // Use pipeline pointer as sentinel (R6)
    if (params._resolvedFor != sig) [[unlikely]] {
      params._resolve(pso->reflection());  // Use new reflection() method
      params._resolvedFor = sig;
    }
    params._apply(*this);
    rawDispatch(gx, gy, gz);
  }

  // Typed compute shader dispatch. ShaderT determines the only accepted
  // Params type at compile time, then delegates to the existing params path.
  template <class ShaderT>
    requires std::derived_from<ShaderT, ComputeShader<ShaderT>>
  void dispatch(const ShaderT& shader, typename ShaderT::Params& params,
                uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    if (!shader.m_pipeline) {
      throw std::runtime_error(
          "cannot dispatch an invalid typed compute shader");
    }
    dispatch(shader.m_pipeline, params, gx, gy, gz);
  }

  // Compatibility overload: plain (gx, gy, gz) → rawDispatch. Picked by
  // overload resolution when no params object is supplied.
  void dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
    rawDispatch(gx, gy, gz);
  }

  // ============ Graphics (R6) ============
  virtual void beginRenderPass(const RenderPassBeginInfo&) = 0;
  virtual void endRenderPass() = 0;

  virtual void bindGraphicsPipeline(PipelineRef pipeline) = 0;

  virtual void setViewport(const Viewport&) = 0;
  virtual void setScissor(const Rect2D&) = 0;

  virtual void bindVertexBuffer(uint32_t binding, BufferRef buf,
                                size_t offset = 0) = 0;
  virtual void bindIndexBuffer(BufferRef buf, IndexFormat fmt,
                               size_t offset = 0) = 0;

  template <class P>
    requires std::is_base_of_v<ShaderParamsBase, P>
  void applyGraphicsParams(PipelineRef pso, P& params) {
    bindGraphicsPipeline(pso);
    const void* sig = pso.get();
    if (params._resolvedFor != sig) [[unlikely]] {
      params._resolve(pso->reflection());
      params._resolvedFor = sig;
    }
    params._apply(*this);
  }

  virtual void draw(uint32_t vertexCount, uint32_t instanceCount = 1,
                    uint32_t firstVertex = 0, uint32_t firstInstance = 0) = 0;
  virtual void drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                           uint32_t firstIndex = 0, int32_t vertexOffset = 0,
                           uint32_t firstInstance = 0) = 0;

  template <class P>
    requires std::is_base_of_v<ShaderParamsBase, P>
  void draw(PipelineRef pso, P& params,
            uint32_t vertexCount, uint32_t instanceCount = 1,
            uint32_t firstVertex = 0, uint32_t firstInstance = 0) {
    applyGraphicsParams(pso, params);
    draw(vertexCount, instanceCount, firstVertex, firstInstance);
  }

  template <class P>
    requires std::is_base_of_v<ShaderParamsBase, P>
  void drawIndexed(PipelineRef pso, P& params,
                   uint32_t indexCount, uint32_t instanceCount = 1,
                   uint32_t firstIndex = 0, int32_t vertexOffset = 0,
                   uint32_t firstInstance = 0) {
    applyGraphicsParams(pso, params);
    drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset,
                firstInstance);
  }

  // ============ End recording ============
  virtual void end() = 0;

 protected:
  CommandList() = default;
};

// ---- ShaderParamsBase::_apply — defined here, after CommandList is
//      complete, since the implementation calls CommandList's mid-tier
//      methods (bindBufferAt / bindImageAt / bindSamplerAt / pushAt). The
//      function is non-template, marked inline to allow the definition to
//      appear in multiple TUs without ODR conflict.
//
// Values are read from centralised storage (_refSlots / _scalarData) by
// slot index. No pointer arithmetic, no reinterpret_cast, no offsetof.
inline void ShaderParamsBase::_apply(CommandList& cmd) const {
  for (const auto& e : _bindings) {
    switch (e.kind) {
      case detail::FieldKind::UAVBuffer:
      case detail::FieldKind::SRVBuffer: {
        const auto& buf = std::get<BufferRef>(
            _refSlots[e.slotIndex].value);
        cmd.bindBufferAt(e.set, e.binding, buf);
        break;
      }
      case detail::FieldKind::UAVImage: {
        const auto& img = std::get<ImageRef>(
            _refSlots[e.slotIndex].value);
        cmd.bindImageAt(e.set, e.binding, img, /*sampler=*/SamplerRef{});
        break;
      }
      case detail::FieldKind::SampledImage: {
        const auto& ib = std::get<ImageBinding>(
            _refSlots[e.slotIndex].value);
        cmd.bindImageAt(e.set, e.binding, ib.image, ib.sampler);
        break;
      }
      case detail::FieldKind::Sampler: {
        const auto& smp = std::get<SamplerRef>(
            _refSlots[e.slotIndex].value);
        cmd.bindSamplerAt(e.set, e.binding, smp);
        break;
      }
      case detail::FieldKind::Scalar: {
        cmd.pushAt(e.pcOffset,
                   _scalarData.data() + e.slotIndex,
                   e.pcSize);
        break;
      }
    }
  }
}

}  // namespace ksk::rhi
