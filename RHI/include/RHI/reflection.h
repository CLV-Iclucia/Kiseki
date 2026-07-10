//
// reflection.h
// Backend-neutral shader reflection result + free function reflectSpirv().
// See docs/rhi-plan.md §3.4.1 / §3.4.3 / §5.2.
//
// Why a free function (not a method on Shader)?
//   • R3 ShaderCompiler is decoupled from Device, so reflection of pure
//     bytecode also wants to be Device-free — it's a CPU-only transform from
//     SPIR-V → ReflectionInfo.
//   • Device::createShader (R4) calls reflectSpirv() under the hood and packs
//     the result into the concrete VulkanShader.
//   • DX12 backend will provide a sibling reflectDxil() in
//     RHI/src/dx12/reflect-dxil.cc that fills the *exact same* ReflectionInfo
//     using IDxcContainerReflection — Pipeline creation logic stays identical
//     across backends (plan §3.4.3).
//
// What's intentionally NOT here:
//   • Specialization constants. Recorded as future work; populated by R5+ when
//     spec_constants land in ShaderCompileOptions.
//   • Workgroup size. SPIRV-Reflect can fish it out of the OpExecutionMode but
//     R4 doesn't need it yet — Pipeline creation passes through to DXC's
//     [numthreads(...)] in the SPIR-V.
//   • Vertex input layout. Plan §3.4.2 GraphicsPipelineDesc carries vertex
//     layout explicitly because SPIR-V doesn't preserve binding-stride —
//     reflection only knows attribute locations + formats.
//

#pragma once

#include <RHI/shader.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ksk::rhi {

// One entry per descriptor binding declared in the shader. Multiple stages
// (vertex + fragment) can list the same (set, binding) — reflection is per
// stage; the Pipeline merges them.
struct DescriptorBindingInfo {
  enum class Type : uint32_t {
    StorageBuffer,   // RWStructuredBuffer / RWByteAddressBuffer (HLSL UAV buffer)
    UniformBuffer,   // cbuffer
    StorageImage,    // RWTexture* (UAV)
    SampledImage,    // Texture* (SRV) — sampler is independent unless combined
    Sampler,         // SamplerState / SamplerComparisonState
  };

  uint32_t set = 0;
  uint32_t binding = 0;
  Type type = Type::StorageBuffer;
  uint32_t count = 1;     // Descriptor array length; 1 for scalar declarations.
  std::string name;       // HLSL/GLSL variable name. Stable for SHADER_PARAMS
                          // name-based binding (R4 §3.4.4); also handy for
                          // diagnostics.
};

// One entry per push-constant block. Vulkan allows multiple ranges per stage
// in principle; HLSL via DXC always emits a single block but we keep the
// vector form for forward compat.
struct PushConstantInfo {
  uint32_t offset = 0;    // Within the stage's PC space.
  uint32_t size = 0;      // Bytes.
};

struct ReflectionInfo {
  ShaderStage stage = ShaderStage::Compute;
  std::string entryPoint;
  std::vector<DescriptorBindingInfo> bindings;
  std::vector<PushConstantInfo> pushConstants;

  // Future:
  //   std::vector<SpecializationConstantInfo> specConstants;
  //   std::array<uint32_t, 3> localSize;  // [numthreads(x,y,z)] for compute
};

// Reflect a SPIR-V bytecode blob. Uses Khronos SPIRV-Reflect under the hood.
//
// Inputs:
//   spirv      — full SPIR-V module (4-byte magic 0x07230203 in word 0).
//                Span over std::byte to match ShaderCompiler::CompiledShader.
//   stage      — caller-asserted stage; we cross-check against the entry-point
//                execution model and log a warning on mismatch (rare with
//                DXC, defensive against hand-rolled bytecode).
//   entryPoint — name to reflect against. SPIR-V supports multiple entry
//                points per module; DXC default is "main".
//
// Returns a fully populated ReflectionInfo. If the module is malformed or the
// entry point is missing, returns a ReflectionInfo with empty `bindings` /
// `pushConstants` and logs the SPIRV-Reflect error via spdlog::error.
//
// CPU-only; safe to call from any thread. SPIRV-Reflect itself does no I/O.
ReflectionInfo reflectSpirv(std::span<const std::byte> spirv,
                            ShaderStage stage,
                            std::string_view entryPoint);

}  // namespace ksk::rhi
