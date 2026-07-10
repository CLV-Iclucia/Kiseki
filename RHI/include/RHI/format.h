//
// format.h
// Backend-neutral pixel / vertex format enum.
// See docs/rhi-plan.md §13.5.
//

#pragma once

#include <cstdint>

namespace ksk::rhi {

enum class Format : uint32_t {
  Undefined = 0,

  // ---- 8-bit ---------------------------------------------------------------
  R8_UNorm,
  R8_SNorm,
  R8_UInt,
  R8_SInt,
  RG8_UNorm,
  RG8_SNorm,
  RG8_UInt,
  RG8_SInt,
  RGBA8_UNorm,
  RGBA8_SNorm,
  RGBA8_UInt,
  RGBA8_SInt,
  RGBA8_UNorm_sRGB,
  BGRA8_UNorm,
  BGRA8_UNorm_sRGB,

  // ---- 16-bit --------------------------------------------------------------
  R16_UNorm,
  R16_UInt,
  R16_SInt,
  R16_Float,
  RG16_UNorm,
  RG16_UInt,
  RG16_SInt,
  RG16_Float,
  RGBA16_UNorm,
  RGBA16_UInt,
  RGBA16_SInt,
  RGBA16_Float,

  // ---- 32-bit --------------------------------------------------------------
  R32_UInt,
  R32_SInt,
  R32_Float,
  RG32_UInt,
  RG32_SInt,
  RG32_Float,
  RGB32_Float,  // Vertex-only on most GPUs (cannot sample as image)
  RGBA32_UInt,
  RGBA32_SInt,
  RGBA32_Float,

  // ---- Packed --------------------------------------------------------------
  R10G10B10A2_UNorm,
  R11G11B10_Float,

  // ---- Depth / stencil -----------------------------------------------------
  D16_UNorm,
  D24_UNorm_S8_UInt,
  D32_Float,
  D32_Float_S8_UInt,

  // ---- Block-compressed ----------------------------------------------------
  BC1_RGBA_UNorm,
  BC3_RGBA_UNorm,
  BC5_RG_UNorm,
  BC7_RGBA_UNorm,
};

// Returns the byte size of one texel for non-block formats, or one block for BC formats.
// Returns 0 for Undefined / unknown.
constexpr uint32_t formatTexelSize(Format f) noexcept {
  switch (f) {
    case Format::Undefined:
      return 0;

    case Format::R8_UNorm:
    case Format::R8_SNorm:
    case Format::R8_UInt:
    case Format::R8_SInt:
      return 1;

    case Format::RG8_UNorm:
    case Format::RG8_SNorm:
    case Format::RG8_UInt:
    case Format::RG8_SInt:
    case Format::R16_UNorm:
    case Format::R16_UInt:
    case Format::R16_SInt:
    case Format::R16_Float:
    case Format::D16_UNorm:
      return 2;

    case Format::RGBA8_UNorm:
    case Format::RGBA8_SNorm:
    case Format::RGBA8_UInt:
    case Format::RGBA8_SInt:
    case Format::RGBA8_UNorm_sRGB:
    case Format::BGRA8_UNorm:
    case Format::BGRA8_UNorm_sRGB:
    case Format::RG16_UNorm:
    case Format::RG16_UInt:
    case Format::RG16_SInt:
    case Format::RG16_Float:
    case Format::R32_UInt:
    case Format::R32_SInt:
    case Format::R32_Float:
    case Format::R10G10B10A2_UNorm:
    case Format::R11G11B10_Float:
    case Format::D24_UNorm_S8_UInt:
    case Format::D32_Float:
      return 4;

    case Format::RGBA16_UNorm:
    case Format::RGBA16_UInt:
    case Format::RGBA16_SInt:
    case Format::RGBA16_Float:
    case Format::RG32_UInt:
    case Format::RG32_SInt:
    case Format::RG32_Float:
    case Format::D32_Float_S8_UInt:  // technically 5, but VkFormat uses 8
      return 8;

    case Format::RGB32_Float:
      return 12;

    case Format::RGBA32_UInt:
    case Format::RGBA32_SInt:
    case Format::RGBA32_Float:
      return 16;

    // Block compressed: bytes per 4x4 block
    case Format::BC1_RGBA_UNorm:
      return 8;
    case Format::BC3_RGBA_UNorm:
    case Format::BC5_RG_UNorm:
    case Format::BC7_RGBA_UNorm:
      return 16;
  }
  return 0;
}

constexpr bool formatIsDepth(Format f) noexcept {
  switch (f) {
    case Format::D16_UNorm:
    case Format::D24_UNorm_S8_UInt:
    case Format::D32_Float:
    case Format::D32_Float_S8_UInt:
      return true;
    default:
      return false;
  }
}

constexpr bool formatHasStencil(Format f) noexcept {
  return f == Format::D24_UNorm_S8_UInt || f == Format::D32_Float_S8_UInt;
}

}  // namespace ksk::rhi
