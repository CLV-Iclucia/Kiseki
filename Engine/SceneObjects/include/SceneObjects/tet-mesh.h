#pragma once

#include <Runtime/runtime-scene.h>

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace ksk::scene {

struct TetMeshObject {};

struct TetMaterial {
  double density = 1000.0;
  double youngsModulus = 1.0e6;
  double poissonRatio = 0.45;
};

struct TetMeshDesc {
  std::vector<glm::dvec3> vertices;
  std::vector<glm::dvec3> initialPositions;
  std::vector<glm::dvec3> initialVelocities;
  std::vector<std::array<int, 4>> tets;
  std::vector<std::array<int, 3>> surfaceTriangles;
  std::vector<std::array<int, 2>> surfaceEdges;
  TetMaterial material;
};

struct TetMeshObjectDesc final : runtime::SceneObjectDesc {
  using ObjectType = TetMeshObject;

  explicit TetMeshObjectDesc(std::string tag = {})
      : SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] runtime::ObjectTypeId typeId() const noexcept override
  {
    return runtime::elementTypeId<TetMeshObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "TetMeshObject";
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties()
      const override;

  TetMeshDesc mesh;
};

}  // namespace ksk::scene
