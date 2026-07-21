#pragma once

#include <Runtime/runtime-scene.h>
#include <SceneObjects/geometry-view.h>

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace ksk::scene {

struct TetMeshObject {};
struct ClothMeshObject {};

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

  [[nodiscard]] TetMeshView meshView() const noexcept
  {
    return TetMeshView{
        .vertices = vertices,
        .tets = tets,
        .surface = SurfaceMeshView{
            .vertices = vertices,
            .edges = surfaceEdges,
            .triangles = surfaceTriangles,
        },
    };
  }
};

[[nodiscard]] inline TetMeshView viewOf(const TetMeshDesc& mesh) noexcept
{
  return mesh.meshView();
}

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

struct ClothMaterial {
  double arealDensity = 1.0;
  double thickness = 1.0e-3;
  double stretchStiffness = 1.0e3;
  double damping = 0.0;
};

struct ClothMeshDesc {
  std::vector<glm::dvec3> vertices;
  std::vector<glm::dvec3> initialPositions;
  std::vector<glm::dvec3> initialVelocities;
  std::vector<std::array<int, 3>> triangles;
  std::vector<std::array<int, 2>> edges;
  ClothMaterial material;

  [[nodiscard]] SurfaceMeshView meshView() const noexcept
  {
    return SurfaceMeshView{
        .vertices = vertices,
        .edges = edges,
        .triangles = triangles,
    };
  }
};

[[nodiscard]] inline SurfaceMeshView viewOf(const ClothMeshDesc& mesh) noexcept
{
  return mesh.meshView();
}

struct ClothMeshObjectDesc final : runtime::SceneObjectDesc {
  using ObjectType = ClothMeshObject;

  explicit ClothMeshObjectDesc(std::string tag = {})
      : SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] runtime::ObjectTypeId typeId() const noexcept override
  {
    return runtime::elementTypeId<ClothMeshObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "ClothMeshObject";
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties()
      const override;

  ClothMeshDesc mesh;
};

}  // namespace ksk::scene
