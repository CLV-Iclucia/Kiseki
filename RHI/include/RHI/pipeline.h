//
// pipeline.h
// Pipeline + ComputePipelineDesc. R4 covers compute only; graphics in R6.
// See docs/rhi-plan.md §3.4.2 / §3.4.3.
//
// What's stored on a Pipeline:
//   • Backend pipeline object (VkPipeline) + layout (VkPipelineLayout +
//     N × VkDescriptorSetLayout) — derived from the bound shader's reflection.
//   • A back-reference to the Shader (so CommandList::dispatch can re-fetch
//     reflection for the SHADER_PARAMS flow without going through Device).
//
// What's NOT here:
//   • PipelineCache. The Vulkan backend keeps a private VkPipelineCache to
//     speed up SPIR-V → ISA. Pipeline-deduplication via Device-side
//     std::map<PipelineDesc, PipelineRef> is plan §3.4.2 R22 work; a
//     description-keyed cache is added when graphics pipelines arrive (R6+).
//

#pragma once

#include <Core/properties.h>
#include <RHI/format.h>
#include <RHI/rc-ptr.h>
#include <RHI/shader.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace ksk::rhi {

struct ComputePipelineDesc {
  ShaderRef shader;

  // Future: specialization constants, optional pipeline cache name, etc.

  // Pipeline cache comparison support
  bool operator<(const ComputePipelineDesc& o) const;
};

struct GraphicsPipelineDesc {
  ShaderRef vertexShader;
  ShaderRef fragmentShader;

  // Vertex input (must be explicitly declared; SPIR-V reflection can't infer stride)
  struct VertexAttribute {
    uint32_t location;
    Format format;
    uint32_t offset;

    bool operator<(const VertexAttribute& o) const {
      if (location != o.location) return location < o.location;
      if (format != o.format) return format < o.format;
      return offset < o.offset;
    }
  };
  struct VertexBinding {
    uint32_t binding;
    uint32_t stride;
    using Attributes = std::vector<VertexAttribute>;
    Attributes attributes;

    bool operator<(const VertexBinding& o) const {
      if (binding != o.binding) return binding < o.binding;
      if (stride != o.stride) return stride < o.stride;
      return attributes < o.attributes;
    }
  };
  using Bindings = std::vector<VertexBinding>;
  Bindings vertexBindings;

  // Rendering state (simplified, sufficient for fluid visualization)
  enum class PrimitiveTopology { TriangleList, LineList, PointList };
  PrimitiveTopology topology = PrimitiveTopology::TriangleList;

  bool depthTest = true;
  bool depthWrite = true;

  // Color blending (simplified: global alpha blend toggle)
  bool alphaBlend = false;

  // Render target format description
  using Formats = std::vector<Format>;
  Formats colorFormats;
  Format depthFormat = Format::Undefined;

  // Pipeline cache comparison support
  bool operator<(const GraphicsPipelineDesc& o) const;
};

  struct RayTracingPipelineDesc
  {

  };

// ComputePipelineDesc comparison (simple: just shader pointer)
inline bool ComputePipelineDesc::operator<(const ComputePipelineDesc& o) const {
  return shader.get() < o.shader.get();
}

// GraphicsPipelineDesc comparison (see plan §1b cookbook)
inline bool GraphicsPipelineDesc::operator<(const GraphicsPipelineDesc& o) const {
  // 1. Compare shader pointers (fastest differentiation)
  if (vertexShader.get() != o.vertexShader.get())
    return vertexShader.get() < o.vertexShader.get();
  if (fragmentShader.get() != o.fragmentShader.get())
    return fragmentShader.get() < o.fragmentShader.get();
  // 2. Compare topology + state flags
  if (topology != o.topology) return topology < o.topology;
  if (depthTest != o.depthTest) return depthTest < o.depthTest;
  if (depthWrite != o.depthWrite) return depthWrite < o.depthWrite;
  if (alphaBlend != o.alphaBlend) return alphaBlend < o.alphaBlend;
  // 3. Compare render target formats
  if (colorFormats != o.colorFormats) return colorFormats < o.colorFormats;
  if (depthFormat != o.depthFormat) return depthFormat < o.depthFormat;
  // 4. Compare vertex layout (last, as shader pointer usually differentiates)
  return vertexBindings < o.vertexBindings;
}

class Pipeline : public core::NonCopyable {
 public:
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  virtual ~Pipeline() = default;

  // R6 new: Return merged reflection info (compute: single shader;
  // graphics: VS ∪ FS)
  virtual const class ReflectionInfo& reflection() const = 0;

  // R6 deprecated: Remove shader(). Reason: undefined for graphics/ray tracing
  // pipelines. Use reflection() instead for SHADER_PARAMS resolution.
  // See docs/rhi-r6-plan.md §3 design decisions.

 protected:
  Pipeline() = default;
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using PipelineRef = RcPtr<Pipeline>;

}  // namespace ksk::rhi
