//
// device.h
// Device interface (R0–R2 surface only).
// See docs/rhi-plan.md §3.1.
//

#pragma once

#include <Core/properties.h>
#include <RHI/backend.h>
#include <RHI/buffer.h>
#include <RHI/commands.h>
#include <RHI/image.h>
#include <RHI/pipeline.h>
#include <RHI/shader.h>
#include <RHI/swapchain.h>
#include <RHI/sync.h>

#include <cstddef>
#include <memory>
#include <span>

namespace sim::rhi {

class ShaderCompiler;  // forward decl (defined in shader-compiler.h)

class Device : public sim::core::NonCopyable {
 public:
  // Returns nullptr if no compatible Vulkan device is available
  // (driver missing, no compatible GPU, validation layer not installed when
  // requested, etc.). Errors are logged via spdlog.
  static std::unique_ptr<Device> create(const DeviceDesc& desc);

  virtual ~Device() = default;

  // ---- Resource creation (all return XxxRef = RcPtr<Xxx>) --------------
  virtual BufferRef createBuffer(const BufferDesc& desc) = 0;
  virtual ImageRef createImage(const ImageDesc& desc) = 0;
  virtual SamplerRef createSampler(const SamplerDesc& desc) = 0;

  // ---- Shaders & pipelines (R4) ----------------------------------------
  // Pure-bytecode entry: caller has already used ShaderCompiler (or has
  // build-time embedded SPIR-V). The shader runs SPIRV-Reflect once and
  // exposes the result via Shader::reflection(); no later refresh.
  virtual ShaderRef createShader(std::span<const std::byte> bytecode,
                                 ShaderStage stage,
                                 std::string_view entryPoint = "main") = 0;

  // Compute pipeline. Layout is derived from the shader's reflection — the
  // caller never writes layout descriptions directly. Plan §3.4.2: pipeline
  // dedupe via Device-side cache is R6 work; for now each call creates a
  // fresh VkPipeline.
  virtual PipelineRef createComputePipeline(const ComputePipelineDesc& desc) = 0;

  // Graphics pipeline (R6). Layout is derived from VS+FS reflection.
  // Pipeline deduplication via Device-side cache.
  virtual PipelineRef createGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;

  // ---- Swapchain (R7) ---------------------------------------------------
  // Creates a swapchain bound to the window surface produced by
  // desc.surfaceCreator. The Device owns the VkSurfaceKHR lifetime.
  // Returns nullptr on failure (e.g. surface creation failed, or the
  // graphics queue doesn't support presentation).
  virtual std::unique_ptr<Swapchain> createSwapchain(const SwapchainDesc& desc) = 0;

  // ---- Sync primitives (unique-ownership; not ref-counted) -------------
  virtual std::unique_ptr<Fence> createFence() = 0;
  virtual std::unique_ptr<Semaphore> createSemaphore() = 0;
  virtual void waitFence(Fence& fence) = 0;
  virtual bool isFenceSignaled(Fence& fence) = 0;
  virtual void resetFence(Fence& fence) = 0;

  // ---- Command recording / submission ----------------------------------
  virtual std::unique_ptr<CommandList> beginCommands(QueueType queue) = 0;

  // submit() borrows wait/signal semaphores; ownership is unchanged.
  // onComplete (optional) is signaled when the GPU finishes this submission.
  virtual void submit(CommandList& cmd,
                      std::span<Semaphore* const> waitSemaphores = {},
                      std::span<Semaphore* const> signalSemaphores = {},
                      Fence* onComplete = nullptr) = 0;

  // Convenience: submit + wait until GPU is idle on the chosen queue.
  // Useful in R0–R2 tests where Tier 0 sync semantics dominate.
  virtual void submitAndWait(CommandList& cmd, QueueType queue) = 0;

  // ---- Introspection ----------------------------------------------------
  virtual Backend backend() const = 0;

  // True once the device is bound to a swapchain (Tier 1 path active).
  // Always false in R0–R6 since we use explicit sync model.
  virtual bool frameLoopActive() const = 0;

  // ---- Shader compilation (R3) -------------------------------------------
  // Convenience: creates a ShaderCompiler pre-bound to this device's backend.
  // The returned compiler will default to this device's backend for all
  // subsequent compileHlsl/compileHlslFile calls, so call sites don't need to
  // repeat targetBackend in ShaderCompileOptions.
  std::unique_ptr<ShaderCompiler> createShaderCompiler() const;

  // ---- Convenience shader creation shortcuts (R4) --------------------------
  ShaderRef createVertexShader(std::span<const std::byte> bytecode, std::string_view entryPoint = "main")
  {
      return createShader(bytecode, ShaderStage::Vertex, entryPoint);
  }

  ShaderRef createFragmentShader(std::span<const std::byte> bytecode, std::string_view entryPoint = "main")
  {
      return createShader(bytecode, ShaderStage::Fragment, entryPoint);
  }
  ShaderRef createComputeShader(std::span<const std::byte> bytecode, std::string_view entryPoint = "main")
  {
      return createShader(bytecode, ShaderStage::Compute, entryPoint);
  }

};

}  // namespace sim::rhi
