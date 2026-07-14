#pragma once

#include <Runtime/runtime-scene.h>
#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::scene {

struct HairObject {};

using HairBlock = glm::dvec4;

struct HairMaterial {
  double density = 1300.0;
  double radius = 4e-5;
  double youngsModulus = 4e9;
  double shearModulus = 1.5e9;
};

struct HairObjectDesc : runtime::SceneObjectDesc {
  using ObjectType = HairObject;

  explicit HairObjectDesc(std::string tag = {})
      : SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] runtime::ObjectTypeId typeId() const noexcept override
  {
    return runtime::elementTypeId<HairObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "HairObject";
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties() const override
  {
    return {
        {"material.density", runtime::PropertyType::Scalar, 1, 1, "Hair density"},
        {"material.radius", runtime::PropertyType::Scalar, 1, 1, "Hair radius"},
        {"material.youngsModulus", runtime::PropertyType::Scalar, 1, 1,
         "Hair Young's modulus"},
        {"material.shearModulus", runtime::PropertyType::Scalar, 1, 1,
         "Hair shear modulus"},
    };
  }

  [[nodiscard]] std::vector<runtime::ObjectRef> rodRefs(
      const runtime::RuntimeSceneDesc& scene) const;

  HairMaterial material;
  std::vector<HairBlock> restBlocks;
};

struct HairStrandDesc {
  std::vector<HairBlock> restBlocks;
  HairMaterial material;
};

[[nodiscard]] runtime::ObjectRef addHair(runtime::RuntimeSceneDesc& scene,
                                         HairStrandDesc hair);
void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef rod,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target);

}  // namespace ksk::scene
