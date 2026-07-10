#pragma once
#include <Core/core.h>
#include <span>
#include <string_view>

namespace ksk::renderer {

/// Data descriptor for a triangle mesh to be rendered.
struct MeshData {
  std::span<const core::Vec3f> positions;
  std::span<const core::Vec3u> triangles;
  std::span<const core::Vec3f> normals;  // may be empty
};

/// Data descriptor for a particle system to be rendered.
struct ParticleData {
  std::span<const core::Vec3f> positions;
  float radius = 0.01f;
};

/// Data descriptor for a scalar field visualization (future use).
struct FieldData {
  std::span<const float> values;
  core::Vec3i resolution;
};

} // namespace ksk::renderer
