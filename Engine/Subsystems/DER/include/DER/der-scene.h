#pragma once

#include <DER/rod.h>
#include <Runtime/runtime-scene.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::der {

struct RodObject {};

struct DERRodDesc {
  std::vector<RodBlock> restBlocks;
  RodMaterial material;
};

struct RodObjectDesc final : runtime::SceneObjectDesc {
  using ObjectType = RodObject;

  explicit RodObjectDesc(std::string tag = {})
      : runtime::SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] runtime::ObjectTypeId typeId() const noexcept override
  {
    return runtime::elementTypeId<RodObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "DERRodObject";
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties() const override
  {
    const int vertexCount = static_cast<int>(restBlocks.size());
    const int twistCount = vertexCount > 0 ? vertexCount - 1 : 0;
    return {
        {"x", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex x-position samples"},
        {"y", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex y-position samples"},
        {"z", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex z-position samples"},
        {"twist", runtime::PropertyType::Scalar, 1, twistCount,
         "Rod edge twist samples"},
    };
  }

  std::vector<RodBlock> restBlocks;
  RodMaterial material;
};

runtime::ObjectRef addRod(runtime::RuntimeSceneDesc& scene, DERRodDesc rod);
void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef rod,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target);

}  // namespace ksk::der
