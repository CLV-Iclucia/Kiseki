#pragma once

#include <array>
#include <span>

#include <glm/glm.hpp>

namespace ksk::scene {

struct SurfaceMeshView {
  std::span<const glm::dvec3> vertices;
  std::span<const std::array<int, 2>> edges;
  std::span<const std::array<int, 3>> triangles;
};

struct TetMeshView {
  std::span<const glm::dvec3> vertices;
  std::span<const std::array<int, 4>> tets;
  SurfaceMeshView surface;
};

}  // namespace ksk::scene
