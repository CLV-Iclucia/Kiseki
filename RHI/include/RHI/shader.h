//
// shader.h
// Shader stage enum + Shader resource class.
// See docs/rhi-plan.md §3.4.1 / §3.4.2.
//
// The Shader class is the GPU-side resource handle. It owns:
//   • Backend bytecode module (VkShaderModule on Vulkan; ID3DBlob on DX12).
//   • A ReflectionInfo built once at creation (no lazy reflection).
//
// Decoupling from Device: like ShaderCompiler, Shader's reflection API is a
// pure-CPU read of cached results, so reflection() is callable without a live
// device queue. The Vulkan-specific module lifetime is tied to Device via the
// 6-line refcount destroy() path (Tier 0 sync — vkDeviceWaitIdle in destroy).
//
// Header dependency policy: shader.h does NOT include <RHI/reflection.h>; it
// only forward-declares ReflectionInfo. This breaks the cycle between
// shader.h and reflection.h (the latter needs ShaderStage from here). Users
// that want to actually read the returned ReflectionInfo must include
// <RHI/reflection.h> themselves — common consumers (commands.h via
// shader-params.h, vk-shader.cc, reflect-spirv.cc) all do.
//

#pragma once

#include <Core/properties.h>
#include <RHI/rc-ptr.h>

#include <atomic>
#include <cstdint>
#include <string_view>

namespace ksk::rhi {

enum class ShaderStage : uint32_t {
  Compute,
  Vertex,
  Fragment,
};

// Forward-declared so Shader::reflection() can declare a return type without
// pulling in the full ReflectionInfo definition. Fine for declarations; the
// .cc that overrides reflection() must include <RHI/reflection.h>.
struct ReflectionInfo;

class Shader : public ksk::core::NonCopyable {
 public:
  // 6-line refcount boilerplate — plan §3.0 / R21: NEVER replace with CRTP.
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  virtual ~Shader() = default;

  virtual ShaderStage stage() const = 0;
  virtual std::string_view entryPoint() const = 0;

  // Returned reflection lives as long as the Shader does. It's safe to take
  // a const& and stash it on a Pipeline; refcount keeps the source alive.
  // Callers must include <RHI/reflection.h> to actually use the result.
  virtual const ReflectionInfo& reflection() const = 0;

 protected:
  Shader() = default;
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using ShaderRef = RcPtr<Shader>;

}  // namespace ksk::rhi
