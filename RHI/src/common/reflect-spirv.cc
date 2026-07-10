//
// reflect-spirv.cc
// SPIR-V → ReflectionInfo via Khronos SPIRV-Reflect. R4.
// See docs/rhi-plan.md §3.4.1 / §5.2.
//
// Why SPIRV-Reflect (not glslang's reflection, not raw SPIR-V parsing)?
//   • Public domain / Apache 2.0; vendored in vcpkg as `spirv-reflect`.
//   • Reflection-only: no codegen, no validation, no optimisation. Tiny TU
//     footprint (~2 KLOC C, no transitive deps).
//   • Tracks the SPIR-V spec closely; updated alongside Vulkan SDK releases.
//
// Memory model:
//   SpvReflectShaderModule owns all returned SpvReflectDescriptorBinding /
//   SpvReflectBlockVariable structs. Their `name` pointers stay valid only
//   until spvReflectDestroyShaderModule(). We copy strings out into std::string
//   before destroying. The RAII helper `ReflectModule` below guarantees that
//   ordering even on early returns.
//

#include <RHI/reflection.h>

#include <spdlog/spdlog.h>
// vcpkg `spirv-reflect` ships TWO copies of the header:
//   • <vcpkg>/include/spirv_reflect.h  — broken in this layout: line 37 reads
//     `#include "./include/spirv/unified1/spirv.h"`, a path that doesn't exist
//     under vcpkg_installed.
//   • <vcpkg>/include/spirv-reflect/spirv_reflect.h — works: it `#include
//     "spirv.h"` co-located in the same dir.
// MSVC searches the vcpkg root include before per-package subdirs (the order
// is set by the vcpkg toolchain, not by us), so a bare `<spirv_reflect.h>`
// resolves to the broken root copy first. Force the subdir path explicitly.
#include <spirv-reflect/spirv_reflect.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ksk::rhi {
namespace {

// ---- RAII -----------------------------------------------------------------
//
// SpvReflectShaderModule is a C-style POD; we wrap its lifetime so all early
// returns get the matching destroy() call.
class ReflectModule {
 public:
  ReflectModule() = default;
  ReflectModule(const ReflectModule&) = delete;
  ReflectModule& operator=(const ReflectModule&) = delete;
  ~ReflectModule() {
    if (m_initialized) spvReflectDestroyShaderModule(&m_module);
  }

  // Returns SPV_REFLECT_RESULT_SUCCESS on success. Caller is expected to log /
  // bail on any other value.
  SpvReflectResult init(size_t size, const void* code) {
    SpvReflectResult r = spvReflectCreateShaderModule(size, code, &m_module);
    m_initialized = (r == SPV_REFLECT_RESULT_SUCCESS);
    return r;
  }

  const SpvReflectShaderModule* get() const { return &m_module; }
  SpvReflectShaderModule* mut() { return &m_module; }

 private:
  SpvReflectShaderModule m_module{};
  bool m_initialized = false;
};

// ---- Mapping helpers ------------------------------------------------------

// Map SPIRV-Reflect's descriptor type enum to our backend-neutral enum.
// Returns true if the kind is one of the 5 we model in R4 (plan §3.4.1).
// Unsupported kinds (acceleration structures, input attachments, texel
// buffers...) will get a warning and be skipped — they're not on the R4–R7
// path.
bool mapDescriptorType(SpvReflectDescriptorType in,
                       DescriptorBindingInfo::Type& out) {
  switch (in) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
      out = DescriptorBindingInfo::Type::Sampler;
      return true;
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      out = DescriptorBindingInfo::Type::SampledImage;
      return true;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      out = DescriptorBindingInfo::Type::StorageImage;
      return true;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      out = DescriptorBindingInfo::Type::UniformBuffer;
      return true;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      out = DescriptorBindingInfo::Type::StorageBuffer;
      return true;
    default:
      return false;  // Texel buffers, input attachments, AS — not in R4 scope.
  }
}

const char* descriptorTypeName(SpvReflectDescriptorType t) {
  switch (t) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER: return "Sampler";
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return "CombinedImageSampler";
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "SampledImage";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "StorageImage";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      return "UniformTexelBuffer";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      return "StorageTexelBuffer";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UniformBuffer";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "StorageBuffer";
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      return "UniformBufferDynamic";
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return "StorageBufferDynamic";
    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return "InputAttachment";
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      return "AccelerationStructureKHR";
  }
  return "<unknown>";
}

// Compute the descriptor count from a SPIRV-Reflect array view.
//
// For non-array bindings, `dim_count == 0` and we use 1.
// For arrays, `dims[i]` are the lengths along each dimension; the descriptor
// count is the product. Vulkan only supports flat arrays, but DXC sometimes
// emits multi-dim arrays for HLSL `Texture2D arr[2][3]` (becomes 6 in SPIR-V).
uint32_t computeDescriptorCount(const SpvReflectBindingArrayTraits& a) {
  if (a.dims_count == 0) return 1u;
  uint32_t total = 1u;
  for (uint32_t i = 0; i < a.dims_count; ++i) {
    // Runtime-sized arrays (`buffer[]`) report 0; treat as 1 for sizing the
    // R4 layout — the descriptor allocator phase will revisit if needed.
    total *= (a.dims[i] == 0u ? 1u : a.dims[i]);
  }
  return total;
}

// Find the SpvReflectEntryPoint matching `name`. SPIRV-Reflect doesn't expose
// a lookup helper; we walk the small array.
const SpvReflectEntryPoint* findEntryPoint(const SpvReflectShaderModule& mod,
                                           std::string_view name) {
  for (uint32_t i = 0; i < mod.entry_point_count; ++i) {
    const auto& ep = mod.entry_points[i];
    if (ep.name && name == ep.name) return &ep;
  }
  return nullptr;
}

// Crosscheck the caller-asserted stage against the entry-point execution model.
// Mismatch is logged but not fatal — the reflection is still usable.
void verifyStage(ShaderStage asserted, SpvExecutionModel actual,
                 std::string_view entry) {
  ShaderStage actualStage{};
  switch (actual) {
    case SpvExecutionModelGLCompute:
      actualStage = ShaderStage::Compute;
      break;
    case SpvExecutionModelVertex:
      actualStage = ShaderStage::Vertex;
      break;
    case SpvExecutionModelFragment:
      actualStage = ShaderStage::Fragment;
      break;
    default:
      spdlog::warn(
          "[reflectSpirv] entry '{}' uses execution model {} which is not in "
          "the R4 scope (compute/vertex/fragment)",
          entry, static_cast<int>(actual));
      return;
  }
  if (actualStage != asserted) {
    spdlog::warn(
        "[reflectSpirv] caller asserted stage {} but entry '{}' is actually "
        "stage {} per SPIR-V execution model",
        static_cast<int>(asserted), entry, static_cast<int>(actualStage));
  }
}

}  // namespace

ReflectionInfo reflectSpirv(std::span<const std::byte> spirv,
                            ShaderStage stage,
                            std::string_view entryPoint) {
  ReflectionInfo info;
  info.stage = stage;
  info.entryPoint = std::string(entryPoint);

  if (spirv.empty()) {
    spdlog::error("[reflectSpirv] empty SPIR-V blob");
    return info;
  }

  ReflectModule mod;
  SpvReflectResult r = mod.init(spirv.size(), spirv.data());
  if (r != SPV_REFLECT_RESULT_SUCCESS) {
    spdlog::error(
        "[reflectSpirv] spvReflectCreateShaderModule failed (code={})",
        static_cast<int>(r));
    return info;
  }

  const SpvReflectEntryPoint* ep = findEntryPoint(*mod.get(), entryPoint);
  if (!ep) {
    spdlog::error(
        "[reflectSpirv] entry point '{}' not found in module (has {} entry "
        "points)",
        entryPoint, mod.get()->entry_point_count);
    return info;
  }
  verifyStage(stage, ep->spirv_execution_model, entryPoint);

  // ---- Descriptor bindings ------------------------------------------------
  //
  // Use the entry-point-scoped enumeration so multi-EP modules don't
  // cross-pollute. SPIRV-Reflect filters to the bindings actually referenced
  // by the chosen entry point + its callees.
  uint32_t bindingCount = 0;
  r = spvReflectEnumerateEntryPointDescriptorBindings(
      mod.get(), ep->name, &bindingCount, nullptr);
  if (r != SPV_REFLECT_RESULT_SUCCESS) {
    spdlog::error(
        "[reflectSpirv] EnumerateEntryPointDescriptorBindings(count) failed "
        "(code={})",
        static_cast<int>(r));
    return info;
  }
  std::vector<SpvReflectDescriptorBinding*> bindings(bindingCount, nullptr);
  if (bindingCount > 0) {
    r = spvReflectEnumerateEntryPointDescriptorBindings(
        mod.get(), ep->name, &bindingCount, bindings.data());
    if (r != SPV_REFLECT_RESULT_SUCCESS) {
      spdlog::error(
          "[reflectSpirv] EnumerateEntryPointDescriptorBindings(data) failed "
          "(code={})",
          static_cast<int>(r));
      return info;
    }
  }

  info.bindings.reserve(bindingCount);
  for (auto* b : bindings) {
    if (!b) continue;
    DescriptorBindingInfo::Type kind{};
    if (!mapDescriptorType(b->descriptor_type, kind)) {
      spdlog::warn(
          "[reflectSpirv] entry '{}' binding (set={}, binding={}) uses "
          "descriptor type {} which is outside R4 scope; skipping",
          entryPoint, b->set, b->binding,
          descriptorTypeName(b->descriptor_type));
      continue;
    }
    DescriptorBindingInfo dbi;
    dbi.set = b->set;
    dbi.binding = b->binding;
    dbi.type = kind;
    dbi.count = computeDescriptorCount(b->array);
    if (b->name && *b->name) {
      dbi.name.assign(b->name);
    }
    info.bindings.push_back(std::move(dbi));
  }

  // ---- Push constants -----------------------------------------------------
  uint32_t pcCount = 0;
  r = spvReflectEnumerateEntryPointPushConstantBlocks(
      mod.get(), ep->name, &pcCount, nullptr);
  if (r != SPV_REFLECT_RESULT_SUCCESS) {
    spdlog::error(
        "[reflectSpirv] EnumerateEntryPointPushConstantBlocks(count) failed "
        "(code={})",
        static_cast<int>(r));
    return info;
  }
  std::vector<SpvReflectBlockVariable*> pcBlocks(pcCount, nullptr);
  if (pcCount > 0) {
    r = spvReflectEnumerateEntryPointPushConstantBlocks(
        mod.get(), ep->name, &pcCount, pcBlocks.data());
    if (r != SPV_REFLECT_RESULT_SUCCESS) {
      spdlog::error(
          "[reflectSpirv] EnumerateEntryPointPushConstantBlocks(data) failed "
          "(code={})",
          static_cast<int>(r));
      return info;
    }
  }

  info.pushConstants.reserve(pcCount);
  for (auto* blk : pcBlocks) {
    if (!blk) continue;
    PushConstantInfo pc;
    pc.offset = blk->offset;
    pc.size = blk->size;
    info.pushConstants.push_back(pc);
  }

  return info;
}

}  // namespace ksk::rhi
