#include <Scene/hair.h>

#include <DER/der-scene.h>

#include <string>
#include <utility>

namespace ksk::scene {
namespace {

der::RodMaterial toDERMaterial(const HairMaterial& material)
{
  der::RodMaterial der_material;
  der_material.density = material.density;
  der_material.radius = material.radius;
  der_material.youngsModulus = material.youngsModulus;
  der_material.shearModulus = material.shearModulus;
  return der_material;
}

}  // namespace

std::vector<runtime::ObjectRef> HairObjectDesc::rodRefs(
    const runtime::RuntimeSceneDesc& scene) const
{
  std::vector<runtime::ObjectRef> result;
  result.reserve(children().size());
  for (runtime::ObjectId childId : children()) {
    runtime::ObjectRef child = scene.findObjectById(childId);
    if (child.isValid() && child.isA<der::RodObject>()) {
      result.push_back(child);
    }
  }
  return result;
}

runtime::ObjectRef addHair(runtime::RuntimeSceneDesc& scene,
                           HairStrandDesc hair)
{
  auto hair_desc = std::make_unique<HairObjectDesc>("hair");
  hair_desc->material = hair.material;
  hair_desc->restBlocks = std::move(hair.restBlocks);

  runtime::ObjectRef hairRef = scene.registerObject(std::move(hair_desc));
  HairObjectDesc* hairDesc = scene.findObjectDescAs<HairObjectDesc>(hairRef);
  if (!hairDesc) {
    throw std::runtime_error("failed to locate registered HairObjectDesc");
  }

  der::DERRodDesc rod;
  rod.restBlocks = hairDesc->restBlocks;
  rod.material = toDERMaterial(hairDesc->material);
  runtime::ObjectRef rodRef = der::addRod(scene, std::move(rod));
  hairDesc->addChildObject(rodRef.id);
  return hairRef;
}

void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef rod,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target)
{
  der::addConstraint(scene, rod, std::move(property), sample, stiffness,
                     std::move(target));
}

}  // namespace ksk::scene
