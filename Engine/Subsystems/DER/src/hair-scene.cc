#include <SceneObjects/hair.h>

#include <DER/der-scene.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ksk::scene {

std::vector<runtime::ObjectRef> HairObjectDesc::rodRefs(
    const runtime::RuntimeSceneDesc& scene) const
{
  std::vector<runtime::ObjectRef> result;
  result.reserve(children().size());
  for (runtime::ObjectId childId : children()) {
    runtime::ObjectRef child = scene.findObjectById(childId);
    if (child.isValid() && child.isA<RodObject>()) {
      result.push_back(child);
    }
  }
  return result;
}

runtime::ObjectRef addHair(runtime::RuntimeSceneDesc& scene,
                           HairObjectDesc hair)
{
  auto hair_desc = std::make_unique<HairObjectDesc>(
      hair.getTag().empty() ? "hair" : hair.getTag());
  hair_desc->material = hair.material;
  hair_desc->restBlocks = std::move(hair.restBlocks);

  runtime::ObjectRef hairRef = scene.registerObject(std::move(hair_desc));
  auto hairDesc = scene.findObjectDescAs<HairObjectDesc>(hairRef);
  if (!hairDesc) {
    throw std::runtime_error("failed to locate registered HairObjectDesc");
  }

  der::DERRodDesc rod;
  rod.restBlocks = hairDesc->restBlocks;
  rod.material = {
      hairDesc->material.density,
      hairDesc->material.radius,
      hairDesc->material.youngsModulus,
      hairDesc->material.shearModulus,
  };
  runtime::ObjectRef rodRef = der::addRod(scene, std::move(rod));

  hairDesc = scene.findObjectDescAs<HairObjectDesc>(hairRef);
  if (!hairDesc) {
    throw std::runtime_error("failed to relocate registered HairObjectDesc");
  }
  hairDesc->addChildObject(rodRef.id);
  return hairRef;
}

}  // namespace ksk::scene
