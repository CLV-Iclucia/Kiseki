//
// shader-types.h
// GPU-friendly vector/matrix types for use with SHADER_PARAM_SCALAR/VEC macros.
//
// These types are trivially copyable, have well-defined size and alignment
// matching HLSL's float2/float3/float4/int2/int3/int4/uint2/uint3/uint4,
// and can be used directly as push-constant members.
//
// IMPORTANT: HLSL packing rules for push constants / constant buffers:
//   - float2 / int2 / uint2:  8 bytes, 8-byte aligned
//   - float3 / int3 / uint3: 12 bytes, 16-byte aligned (!)
//   - float4 / int4 / uint4: 16 bytes, 16-byte aligned
//   - mat4:                  64 bytes, 16-byte aligned
//
// When using vec3 types as push constants, be mindful of the 16-byte
// alignment requirement in HLSL. The C++ struct layout must match.
//

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdint>
#include <type_traits>

namespace sim::rhi {

// ============================================================================
// Float vector types (matching HLSL float2/float3/float4)
// ============================================================================
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

// ============================================================================
// Signed integer vector types (matching HLSL int2/int3/int4)
// ============================================================================
using int2 = glm::ivec2;
using int3 = glm::ivec3;
using int4 = glm::ivec4;

// ============================================================================
// Unsigned integer vector types (matching HLSL uint2/uint3/uint4)
// ============================================================================
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;

// ============================================================================
// Matrix types (matching HLSL float2x2/float3x3/float4x4)
// ============================================================================
using float2x2 = glm::mat2;
using float3x3 = glm::mat3;
using float4x4 = glm::mat4;

// ============================================================================
// Type traits: IsVectorType — recognises the GPU vector types above
// ============================================================================
namespace detail {

template <class T>
struct IsGpuVectorType : std::false_type {};

// float vectors
template <> struct IsGpuVectorType<float2> : std::true_type {};
template <> struct IsGpuVectorType<float3> : std::true_type {};
template <> struct IsGpuVectorType<float4> : std::true_type {};

// int vectors
template <> struct IsGpuVectorType<int2> : std::true_type {};
template <> struct IsGpuVectorType<int3> : std::true_type {};
template <> struct IsGpuVectorType<int4> : std::true_type {};

// uint vectors
template <> struct IsGpuVectorType<uint2> : std::true_type {};
template <> struct IsGpuVectorType<uint3> : std::true_type {};
template <> struct IsGpuVectorType<uint4> : std::true_type {};

// matrices
template <> struct IsGpuVectorType<float2x2> : std::true_type {};
template <> struct IsGpuVectorType<float3x3> : std::true_type {};
template <> struct IsGpuVectorType<float4x4> : std::true_type {};

// Generic GLM vec detection (covers any glm::vec<L, T, Q>)
template <glm::length_t L, class T, glm::qualifier Q>
struct IsGpuVectorType<glm::vec<L, T, Q>> : std::true_type {};

// Generic GLM mat detection (covers any glm::mat<C, R, T, Q>)
template <glm::length_t C, glm::length_t R, class T, glm::qualifier Q>
struct IsGpuVectorType<glm::mat<C, R, T, Q>> : std::true_type {};

}  // namespace detail

// Convenience variable template
template <class T>
inline constexpr bool is_gpu_vector_type_v = detail::IsGpuVectorType<T>::value;

// ============================================================================
// Static assertions to verify sizes match HLSL expectations
// ============================================================================
static_assert(sizeof(float2) == 8, "float2 must be 8 bytes");
static_assert(sizeof(float3) == 12, "float3 must be 12 bytes");
static_assert(sizeof(float4) == 16, "float4 must be 16 bytes");
static_assert(sizeof(int2) == 8, "int2 must be 8 bytes");
static_assert(sizeof(int3) == 12, "int3 must be 12 bytes");
static_assert(sizeof(int4) == 16, "int4 must be 16 bytes");
static_assert(sizeof(uint2) == 8, "uint2 must be 8 bytes");
static_assert(sizeof(uint3) == 12, "uint3 must be 12 bytes");
static_assert(sizeof(uint4) == 16, "uint4 must be 16 bytes");
static_assert(sizeof(float4x4) == 64, "float4x4 must be 64 bytes");

// Verify trivially copyable (required for push-constant memcpy)
static_assert(std::is_trivially_copyable_v<float2>);
static_assert(std::is_trivially_copyable_v<float3>);
static_assert(std::is_trivially_copyable_v<float4>);
static_assert(std::is_trivially_copyable_v<int2>);
static_assert(std::is_trivially_copyable_v<int3>);
static_assert(std::is_trivially_copyable_v<int4>);
static_assert(std::is_trivially_copyable_v<uint2>);
static_assert(std::is_trivially_copyable_v<uint3>);
static_assert(std::is_trivially_copyable_v<uint4>);
static_assert(std::is_trivially_copyable_v<float2x2>);
static_assert(std::is_trivially_copyable_v<float3x3>);
static_assert(std::is_trivially_copyable_v<float4x4>);

// ============================================================================
// HLSL/Vulkan push-constant alignment rules
// ============================================================================
//
// Vulkan std430 / push-constant alignment:
//   - scalar (4 bytes):  4-byte aligned
//   - vec2   (8 bytes):  8-byte aligned
//   - vec3   (12 bytes): 16-byte aligned (!)
//   - vec4   (16 bytes): 16-byte aligned
//   - matNxM:            column-major, each column is a vecN → aligned as vecN
//
// gpuAlignOf<T>() returns the alignment that the Vulkan push-constant layout
// requires for type T. The framework uses this to automatically insert padding
// in the scalar data buffer so C++ and HLSL layouts match.
//
namespace detail {

// Default: scalar types align to their own size (4 for float/uint32_t/int32_t)
template <class T>
struct GpuAlignOf {
  static constexpr uint32_t value = static_cast<uint32_t>(sizeof(T) <= 4 ? 4 : sizeof(T));
};

// vec2 types: 8-byte aligned
template <> struct GpuAlignOf<float2>  { static constexpr uint32_t value = 8; };
template <> struct GpuAlignOf<int2>    { static constexpr uint32_t value = 8; };
template <> struct GpuAlignOf<uint2>   { static constexpr uint32_t value = 8; };

// vec3 types: 16-byte aligned (Vulkan spec requirement)
template <> struct GpuAlignOf<float3>  { static constexpr uint32_t value = 16; };
template <> struct GpuAlignOf<int3>    { static constexpr uint32_t value = 16; };
template <> struct GpuAlignOf<uint3>   { static constexpr uint32_t value = 16; };

// vec4 types: 16-byte aligned
template <> struct GpuAlignOf<float4>  { static constexpr uint32_t value = 16; };
template <> struct GpuAlignOf<int4>    { static constexpr uint32_t value = 16; };
template <> struct GpuAlignOf<uint4>   { static constexpr uint32_t value = 16; };

// Matrix types: aligned as their column vector type (16-byte for mat with vec3/vec4 cols)
template <> struct GpuAlignOf<float2x2> { static constexpr uint32_t value = 8; };
template <> struct GpuAlignOf<float3x3> { static constexpr uint32_t value = 16; };
template <> struct GpuAlignOf<float4x4> { static constexpr uint32_t value = 16; };

// Generic GLM vec: align = (L >= 3) ? 16 : L * sizeof(T)
template <glm::length_t L, class T, glm::qualifier Q>
struct GpuAlignOf<glm::vec<L, T, Q>> {
  static constexpr uint32_t value = (L >= 3) ? 16u : static_cast<uint32_t>(L * sizeof(T));
};

// Generic GLM mat: align as the column vector
template <glm::length_t C, glm::length_t R, class T, glm::qualifier Q>
struct GpuAlignOf<glm::mat<C, R, T, Q>> {
  static constexpr uint32_t value = GpuAlignOf<glm::vec<R, T, Q>>::value;
};

}  // namespace detail

template <class T>
inline constexpr uint32_t gpu_align_of_v = detail::GpuAlignOf<T>::value;

// Helper: align a byte offset up to the given alignment
inline constexpr uint32_t alignUp(uint32_t offset, uint32_t alignment) {
  return (offset + alignment - 1u) & ~(alignment - 1u);
}

}  // namespace sim::rhi
