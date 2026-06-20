//
// shader-params.h
// SHADER_PARAMS macro system. Header-only by design.
// See docs/rhi-plan.md §3.4.4 / §13.1 / §13.2.
//
// User-facing usage:
//   SHADER_PARAMS_BEGIN(SaxpyParams)
//     SHADER_PARAM_UAV   (BufferRef, g_y);
//     SHADER_PARAM_SCALAR(uint32_t,  count);
//     SHADER_PARAM_SCALAR(float,     alpha);
//   SHADER_PARAMS_END();
//
//   SaxpyParams params;
//   params.g_y   = y;       // looks like assignment — IS assignment, but
//   params.count = N;       // each field is a ParamSlot<T> wrapper that
//   params.alpha = 2.5f;    // pushes the value into centralised storage.
//   cmd->dispatch(pso, params, divRoundUp(N, 256), 1, 1);
//
// ===== Design intent (the contract this file enforces) =====================
//
// Two non-negotiable properties:
//
//   1. Each field is a wrapper type with an overloaded operator= so users
//      see plain "params.g_y = buf" syntax, but the framework can hook the
//      assignment if it ever needs to (validation, change-tracking, etc.).
//      The wrapper is `ParamSlot<T>` below — it stores the value locally
//      AND pushes it into centralised storage on every assignment.
//
//   2. ZERO static accumulator variables per field. The schema is built
//      per-instance via DMI side effects at construction time.
//
// ===== Internal mechanics ==================================================
//
//   1. SHADER_PARAM_*(Type, FieldName) expands into the class body:
//        a. A field declaration: `ParamSlot<Type> FieldName = (...);`
//        b. The `(...)` is a comma expression whose left operand calls
//           `this->_registerField(name, kind, valueSize)` — populating
//           `_schema` and allocating a slot in _refSlots or _scalarData —
//           and whose right operand constructs the ParamSlot with the
//           base pointer and slot index.
//      Result: every instance, on construction, registers its fields.
//
//   2. ParamSlot<T>::operator= pushes the assigned value into the base
//      class's centralised storage (_refSlots for Ref types, _scalarData
//      for trivially-copyable scalars). No offset arithmetic, no
//      reinterpret_cast, no layout assumptions.
//
//   3. CommandList::dispatch(pso, params, ...) checks the sentinel
//      `params._resolvedFor != pso.get()`; on mismatch it calls
//      `params._resolve(pso->reflection())` to map field names →
//      (set, binding) via the schema.
//
//   4. `params._apply(cmd)` walks `_bindings` (vector<ResolvedEntry>) once
//      and issues `cmd.bindBufferAt / bindImageAt / bindSamplerAt / pushAt`.
//      No reinterpret_cast, no pointer arithmetic — values are read
//      directly from typed centralised storage by slot index.
//
// ===== Push-constant offset strategy (plan §13.2 R23) ======================
//
//   SCALAR fields claim PC bytes in declaration order, starting from the
//   reflected PC block's `offset`. The HLSL `[[vk::push_constant]] struct`
//   field order MUST match the C++ SHADER_PARAM_SCALAR order. Mismatches
//   are runtime errors when total size overflows the reflected block; same-
//   sized misorderings are silently wrong (validation TODO — needs PC
//   member names from reflection, not in scope this round).
//
// ===== Why header-only =====================================================
//
//   `_resolve` and `_apply` are non-template member functions of the non-
//   template `ShaderParamsBase`, so they can in principle live in a .cc.
//   We keep `_resolve` inline here (no CommandList dependency) and define
//   `_apply` inline in commands.h (after CommandList is complete) so the
//   whole RHI module remains header-rich + easy to inline at call sites.
//   No per-class explicit instantiation or schema initialiser is needed.
//

#pragma once

#include <RHI/buffer.h>
#include <RHI/image.h>
#include <RHI/reflection.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace sim::rhi {

// Forward decl — CommandList is referenced only by the inline _apply impl
// in commands.h.
class CommandList;

// Forward decl — ShaderParamsBase needed by ParamSlot.
class ShaderParamsBase;

// ============================================================================
// 1. detail:: schema types + kind helpers
// ============================================================================
namespace detail {

enum class FieldKind : uint8_t {
  UAVBuffer,     // RWStructuredBuffer  → VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  UAVImage,      // RWTexture*          → VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
  SRVBuffer,     // ConstantBuffer/StructuredBuffer → UNIFORM/STORAGE_BUFFER
  SampledImage,  // Texture + Sampler combo → COMBINED_IMAGE_SAMPLER
  Sampler,       // standalone SamplerState
  Scalar,        // push-constant member
};

// Per-field metadata, accumulated into ShaderParamsBase::_schema by the
// SHADER_PARAM_* macro's DMI side effect.
struct FieldInfo {
  const char* name;       // string literal — no ownership
  FieldKind kind;
  uint32_t slotIndex;     // index into _refSlots (for Ref kinds) or byte
                          // offset into _scalarData (for Scalar)
  uint32_t valueSize;     // sizeof of the underlying T (for Scalar push consts)
};

// Per-instance resolved record. Built once on first dispatch (or on
// shader-switch). The hot path consumes this directly, no string ops.
struct ResolvedEntry {
  FieldKind kind = FieldKind::UAVBuffer;
  uint32_t slotIndex = 0;  // index into _refSlots or byte offset in _scalarData

  // For descriptor kinds:
  uint32_t set = 0;
  uint32_t binding = 0;

  // For Scalar:
  uint32_t pcOffset = 0;
  uint32_t pcSize = 0;
};

// ---- Centralised Ref storage ----
// A type-erased slot that can hold any of the Ref/Image binding types.
// Using std::variant gives us type safety without reinterpret_cast.
struct RefSlot {
  std::variant<std::monostate, BufferRef, ImageRef, SamplerRef, ImageBinding>
      value;
};

// Per-macro kind traits — same C++ type can mean different kinds depending
// on which SHADER_PARAM_* macro it's under (BufferRef as UAV vs SRV).
template <class T>
struct UAVKindOf;
template <>
struct UAVKindOf<BufferRef> {
  static constexpr FieldKind kind = FieldKind::UAVBuffer;
};
template <>
struct UAVKindOf<ImageRef> {
  static constexpr FieldKind kind = FieldKind::UAVImage;
};

template <class T>
struct SRVKindOf;
template <>
struct SRVKindOf<BufferRef> {
  static constexpr FieldKind kind = FieldKind::SRVBuffer;
};

template <class T>
struct ImageKindOf;
template <>
struct ImageKindOf<ImageBinding> {
  static constexpr FieldKind kind = FieldKind::SampledImage;
};

template <class T>
struct SamplerKindOf;
template <>
struct SamplerKindOf<SamplerRef> {
  static constexpr FieldKind kind = FieldKind::Sampler;
};

inline bool kindMatchesDescriptorType(FieldKind k,
                                      DescriptorBindingInfo::Type t) {
  using DT = DescriptorBindingInfo::Type;
  switch (k) {
    case FieldKind::UAVBuffer:
      return t == DT::StorageBuffer;
    case FieldKind::UAVImage:
      return t == DT::StorageImage;
    case FieldKind::SRVBuffer:
      return t == DT::UniformBuffer || t == DT::StorageBuffer;
    case FieldKind::SampledImage:
      return t == DT::SampledImage;
    case FieldKind::Sampler:
      return t == DT::Sampler;
    case FieldKind::Scalar:
      return false;
  }
  return false;
}

inline const char* fieldKindToString(FieldKind k) {
  switch (k) {
    case FieldKind::UAVBuffer:    return "UAVBuffer";
    case FieldKind::UAVImage:     return "UAVImage";
    case FieldKind::SRVBuffer:    return "SRVBuffer";
    case FieldKind::SampledImage: return "SampledImage";
    case FieldKind::Sampler:      return "Sampler";
    case FieldKind::Scalar:       return "Scalar";
  }
  return "?";
}

inline const char* descriptorTypeToString(DescriptorBindingInfo::Type t) {
  using DT = DescriptorBindingInfo::Type;
  switch (t) {
    case DT::StorageBuffer: return "StorageBuffer";
    case DT::UniformBuffer: return "UniformBuffer";
    case DT::StorageImage:  return "StorageImage";
    case DT::SampledImage:  return "SampledImage";
    case DT::Sampler:       return "Sampler";
  }
  return "?";
}

}  // namespace detail

// ============================================================================
// 2. ShaderParamsBase — non-template; holds per-instance state + storage
// ============================================================================
//
// This is the only base class for SHADER_PARAMS-generated structs. It carries:
//   - _schema:       per-instance vector of FieldInfo, populated by the
//                    SHADER_PARAM_* macro's DMI side effect at construction.
//   - _bindings:     per-instance resolved bindings, built on first dispatch.
//   - _resolvedFor:  sentinel — which Shader* did we resolve against?
//   - _refSlots:     centralised Ref storage (variant-based, type-safe).
//   - _scalarData:   centralised scalar storage (byte buffer, memcpy'd).
//
// All public members are intentionally `_`-prefixed to mark them as
// framework-managed; SHADER_PARAMS users don't touch them directly.
//
class ShaderParamsBase {
 public:
  std::vector<detail::FieldInfo> _schema;
  std::vector<detail::ResolvedEntry> _bindings;
  const void* _resolvedFor = nullptr;

  // Centralised value storage — written by ParamSlot::operator=,
  // read by _apply. No pointer arithmetic or reinterpret_cast needed.
  std::vector<detail::RefSlot> _refSlots;
  std::vector<std::byte> _scalarData;

  // Walk _schema + reflection, build _bindings. Throws std::runtime_error on
  // mismatch (missing binding, kind/type incompatible, PC overflow). Inline
  // here — no CommandList dependency.
  void _resolve(const ReflectionInfo& ri);

  // Walk _bindings, issue mid-tier CommandList calls. Defined inline in
  // commands.h since it depends on CommandList being a complete type.
  void _apply(CommandList& cmd) const;

  void _invalidate() noexcept {
    _resolvedFor = nullptr;
    _bindings.clear();
  }

  // ---- Storage update API (called by ParamSlot::operator=) ----
  void _setRef(uint32_t slotIndex, const BufferRef& v) {
    _refSlots[slotIndex].value = v;
  }
  void _setRef(uint32_t slotIndex, const ImageRef& v) {
    _refSlots[slotIndex].value = v;
  }
  void _setRef(uint32_t slotIndex, const SamplerRef& v) {
    _refSlots[slotIndex].value = v;
  }
  void _setRef(uint32_t slotIndex, const ImageBinding& v) {
    _refSlots[slotIndex].value = v;
  }
  void _setScalar(uint32_t byteOffset, const void* data, uint32_t size) {
    std::memcpy(_scalarData.data() + byteOffset, data, size);
  }

 protected:
  ShaderParamsBase() = default;
  ~ShaderParamsBase() = default;  // non-virtual; SHADER_PARAMS structs are
                                   // never destroyed via base pointer

  // Called from the SHADER_PARAM_* macro's DMI side effect for Ref fields.
  // Allocates a RefSlot and returns the slot index.
  uint32_t _registerRefField(const char* name, detail::FieldKind k,
                             uint32_t valueSize) {
    uint32_t idx = static_cast<uint32_t>(_refSlots.size());
    _refSlots.emplace_back();
    _schema.push_back({name, k, idx, valueSize});
    return idx;
  }

  // Called from the SHADER_PARAM_* macro's DMI side effect for Scalar fields.
  // Allocates bytes in _scalarData and returns the byte offset.
  uint32_t _registerScalarField(const char* name, uint32_t valueSize) {
    uint32_t offset = static_cast<uint32_t>(_scalarData.size());
    _scalarData.resize(_scalarData.size() + valueSize, std::byte{0});
    _schema.push_back({name, detail::FieldKind::Scalar, offset, valueSize});
    return offset;
  }
};

// ============================================================================
// 3. ParamSlot<T> — field wrapper with transparent operator=
// ============================================================================
//
// ParamSlot holds a local copy of the value AND pushes updates into the
// base class's centralised storage on every assignment. This eliminates
// all offsetof / reinterpret_cast / pointer-arithmetic tricks.
//
// Usage by SHADER_PARAMS users is transparent:
//   ParamSlot<BufferRef> g_y;
//   g_y = someBuffer;           // operator=(const T&)
//   BufferRef ref = g_y;        // implicit conversion via operator const T&
//   if (g_y.get())              // explicit access
//

// ---- RefParamSlot: for Ref types (BufferRef, ImageRef, SamplerRef, ImageBinding)
template <class T>
class RefParamSlot {
 public:
  RefParamSlot() = default;

  // Constructed by macro with base pointer and slot index.
  RefParamSlot(ShaderParamsBase* base, uint32_t slotIndex)
      : m_base(base), m_slotIndex(slotIndex) {}

  RefParamSlot(const RefParamSlot&) = default;
  RefParamSlot(RefParamSlot&&) noexcept = default;
  RefParamSlot& operator=(const RefParamSlot& o) {
    if (this != &o) {
      m_value = o.m_value;
      if (m_base) m_base->_setRef(m_slotIndex, m_value);
    }
    return *this;
  }
  RefParamSlot& operator=(RefParamSlot&& o) noexcept {
    if (this != &o) {
      m_value = std::move(o.m_value);
      if (m_base) m_base->_setRef(m_slotIndex, m_value);
    }
    return *this;
  }

  // ---- The user-facing assignments ----
  RefParamSlot& operator=(const T& v) {
    m_value = v;
    if (m_base) m_base->_setRef(m_slotIndex, m_value);
    return *this;
  }
  RefParamSlot& operator=(T&& v) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    m_value = std::move(v);
    if (m_base) m_base->_setRef(m_slotIndex, m_value);
    return *this;
  }

  // ---- Read access ----
  operator const T&() const noexcept { return m_value; }
  const T& get() const noexcept { return m_value; }
  T& get() noexcept { return m_value; }

 private:
  T m_value{};
  ShaderParamsBase* m_base = nullptr;
  uint32_t m_slotIndex = 0;
};

// ---- ScalarParamSlot: for trivially-copyable types (uint32_t, float, etc.)
template <class T>
class ScalarParamSlot {
 public:
  ScalarParamSlot() = default;

  // Constructed by macro with base pointer and byte offset.
  ScalarParamSlot(ShaderParamsBase* base, uint32_t byteOffset)
      : m_base(base), m_byteOffset(byteOffset) {}

  ScalarParamSlot(const ScalarParamSlot&) = default;
  ScalarParamSlot(ScalarParamSlot&&) noexcept = default;
  ScalarParamSlot& operator=(const ScalarParamSlot& o) {
    if (this != &o) {
      m_value = o.m_value;
      if (m_base) m_base->_setScalar(m_byteOffset, &m_value, sizeof(T));
    }
    return *this;
  }
  ScalarParamSlot& operator=(ScalarParamSlot&& o) noexcept {
    if (this != &o) {
      m_value = std::move(o.m_value);
      if (m_base) m_base->_setScalar(m_byteOffset, &m_value, sizeof(T));
    }
    return *this;
  }

  // ---- The user-facing assignments ----
  ScalarParamSlot& operator=(const T& v) {
    m_value = v;
    if (m_base) m_base->_setScalar(m_byteOffset, &m_value, sizeof(T));
    return *this;
  }
  ScalarParamSlot& operator=(T&& v) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    m_value = std::move(v);
    if (m_base) m_base->_setScalar(m_byteOffset, &m_value, sizeof(T));
    return *this;
  }

  // ---- Read access ----
  operator const T&() const noexcept { return m_value; }
  const T& get() const noexcept { return m_value; }
  T& get() noexcept { return m_value; }

 private:
  T m_value{};
  ShaderParamsBase* m_base = nullptr;
  uint32_t m_byteOffset = 0;
};

// ---- ParamSlot type alias: selects RefParamSlot or ScalarParamSlot ----
// Ref types: BufferRef, ImageRef, SamplerRef, ImageBinding
// Scalar types: everything else (trivially copyable)
namespace detail {
template <class T>
struct IsRefType : std::false_type {};
template <>
struct IsRefType<BufferRef> : std::true_type {};
template <>
struct IsRefType<ImageRef> : std::true_type {};
template <>
struct IsRefType<SamplerRef> : std::true_type {};
template <>
struct IsRefType<ImageBinding> : std::true_type {};
}  // namespace detail

template <class T>
using ParamSlot = std::conditional_t<detail::IsRefType<T>::value,
                                     RefParamSlot<T>,
                                     ScalarParamSlot<T>>;

// ============================================================================
// 4. ShaderParamsBase::_resolve — non-template, inline definition
// ============================================================================
inline void ShaderParamsBase::_resolve(const ReflectionInfo& ri) {
  // Build name → DescriptorBindingInfo* lookup. Single pass over ri so total
  // work is O(N + M).
  std::unordered_map<std::string_view, const DescriptorBindingInfo*> byName;
  byName.reserve(ri.bindings.size());
  for (const auto& b : ri.bindings) byName.emplace(b.name, &b);

  _bindings.clear();
  _bindings.reserve(_schema.size());

  // PC block accounting (plan §13.2 R23: at most one PC block in HLSL/DXC).
  const uint32_t pcBlockOffset =
      ri.pushConstants.empty() ? 0u : ri.pushConstants[0].offset;
  const uint32_t pcBlockSize =
      ri.pushConstants.empty() ? 0u : ri.pushConstants[0].size;
  uint32_t pcRunning = 0;

  for (const auto& f : _schema) {
    detail::ResolvedEntry e{};
    e.kind = f.kind;
    e.slotIndex = f.slotIndex;

    if (f.kind == detail::FieldKind::Scalar) {
      if (pcRunning + f.valueSize > pcBlockSize) {
        throw std::runtime_error(
            std::string("shader param '") + f.name +
            "': scalar params total size (" +
            std::to_string(pcRunning + f.valueSize) +
            ") exceeds push constant block size (" +
            std::to_string(pcBlockSize) +
            " bytes). Check HLSL [[vk::push_constant]] struct matches C++ "
            "SHADER_PARAM_SCALAR declaration order and types.");
      }
      e.pcOffset = pcBlockOffset + pcRunning;
      e.pcSize = f.valueSize;
      pcRunning += f.valueSize;
    } else {
      auto it = byName.find(f.name);
      if (it == byName.end()) {
        std::string available;
        for (const auto& b : ri.bindings) {
          available += "\n  - " + b.name;
        }
        throw std::runtime_error(
            std::string("shader param '") + f.name +
            "' not found in shader reflection. Available bindings:" +
            available);
      }
      const auto* b = it->second;
      if (!detail::kindMatchesDescriptorType(f.kind, b->type)) {
        throw std::runtime_error(
            std::string("shader param '") + f.name + "': declared as " +
            detail::fieldKindToString(f.kind) +
            " in C++ but shader reflection says " +
            detail::descriptorTypeToString(b->type));
      }
      e.set = b->set;
      e.binding = b->binding;
    }

    _bindings.push_back(e);
  }
}

}  // namespace sim::rhi

// ============================================================================
// 5. User-facing macros
// ============================================================================
//
// MSVC needs /Zc:preprocessor to expand `#FieldName` consistently — the
// project already enables this in RHI/CMakeLists.txt.
//
// Each SHADER_PARAM_REF expands to:
//   ParamSlot<Type> FieldName =
//       ::sim::rhi::ParamSlot<Type>(this, this->_registerRefField(...));
//
// Each SHADER_PARAM_SCALAR expands to:
//   ParamSlot<Type> FieldName =
//       ::sim::rhi::ParamSlot<Type>(this, this->_registerScalarField(...));
//
// The DMI calls _registerRefField/_registerScalarField on the partially-
// constructed leaf. The base sub-object is fully constructed by the time
// members are initialised, so _refSlots/_scalarData vectors are live.
//
// NO offsetof. NO reinterpret_cast. NO pointer arithmetic for field access.
//

#define SHADER_PARAMS_BEGIN(Name)                                              \
  struct Name : public ::sim::rhi::ShaderParamsBase {                          \
    using _Self = Name;

#define SHADER_PARAM_UAV(Type, FieldName)                                      \
  ::sim::rhi::ParamSlot<Type> FieldName{                                       \
      this,                                                                    \
      this->_registerRefField(                                                 \
          #FieldName,                                                          \
          ::sim::rhi::detail::UAVKindOf<Type>::kind,                           \
          static_cast<uint32_t>(sizeof(Type)))}

#define SHADER_PARAM_SRV(Type, FieldName)                                      \
  ::sim::rhi::ParamSlot<Type> FieldName{                                       \
      this,                                                                    \
      this->_registerRefField(                                                 \
          #FieldName,                                                          \
          ::sim::rhi::detail::SRVKindOf<Type>::kind,                           \
          static_cast<uint32_t>(sizeof(Type)))}

#define SHADER_PARAM_IMAGE(Type, FieldName)                                    \
  ::sim::rhi::ParamSlot<Type> FieldName{                                       \
      this,                                                                    \
      this->_registerRefField(                                                 \
          #FieldName,                                                          \
          ::sim::rhi::detail::ImageKindOf<Type>::kind,                         \
          static_cast<uint32_t>(sizeof(Type)))}

#define SHADER_PARAM_SAMPLER(Type, FieldName)                                  \
  ::sim::rhi::ParamSlot<Type> FieldName{                                       \
      this,                                                                    \
      this->_registerRefField(                                                 \
          #FieldName,                                                          \
          ::sim::rhi::detail::SamplerKindOf<Type>::kind,                       \
          static_cast<uint32_t>(sizeof(Type)))}

#define SHADER_PARAM_SCALAR(Type, FieldName)                                   \
  static_assert(::std::is_trivially_copyable_v<Type>,                          \
                "SHADER_PARAM_SCALAR(" #Type ", " #FieldName                   \
                "): Type must be trivially copyable");                         \
  ::sim::rhi::ParamSlot<Type> FieldName{                                       \
      this,                                                                    \
      this->_registerScalarField(                                              \
          #FieldName,                                                          \
          static_cast<uint32_t>(sizeof(Type)))}

#define SHADER_PARAMS_END()                                                    \
  }
