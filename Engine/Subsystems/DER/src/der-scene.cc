#include <DER/der-scene.h>

#include <DER/der-subsystem.h>

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ksk::der {

namespace {

class DERSubsystemDesc final : public runtime::SubsystemDesc {
 public:
  std::vector<runtime::ObjectId> rods;

  void addRod(runtime::ObjectId rod)
  {
    rods.push_back(rod);
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "der";
  }

  void build(runtime::SubsystemBuildContext& context) const override;
};

DERSubsystemDesc* findDERSubsystem(runtime::RuntimeSceneDesc& scene)
{
  for (const auto& subsystem : scene.subsystems) {
    if (subsystem && subsystem->typeName() == "der") {
      return static_cast<DERSubsystemDesc*>(subsystem.get());
    }
  }
  return nullptr;
}

}  // namespace

void DERSubsystemDesc::build(runtime::SubsystemBuildContext& context) const
{
  if (rods.empty()) {
    throw std::runtime_error("DER subsystem requires at least one rod");
  }

  std::vector<Rod> runtime_rods;
  runtime_rods.reserve(rods.size());
  std::vector<DERConstraintBinding> runtime_constraints;
  const runtime::RuntimeSceneDesc& scene = context.sceneDesc();
  for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
       ++rod_index) {
    const runtime::ObjectId rodId = rods[rod_index];
    runtime::ObjectRef rodRef = scene.findObjectById(rodId);
    const RodObjectDesc* rodDesc = scene.findObjectDescAs<RodObjectDesc>(rodRef);
    if (!rodDesc) {
      throw std::runtime_error("DER subsystem references a missing rod object");
    }
    runtime_rods.emplace_back(rodDesc->restBlocks, rodDesc->material);

    for (const runtime::SceneConstraintDesc& constraint : scene.constraints) {
      if (constraint.object.id == rodId) {
        runtime_constraints.push_back(DERConstraintBinding{
            .rod = rod_index,
            .constraint = constraint,
        });
      }
    }
  }

  context.addSubsystem(
      std::make_unique<DERSubsystem>(context.nextSubsystemId(),
                                     std::move(runtime_rods),
                                     std::move(runtime_constraints),
                                     context.nextScalarOffset(),
                                     context.gravity()),
      std::string(typeName()));
}

runtime::ObjectRef addRod(runtime::RuntimeSceneDesc& scene, DERRodDesc rod)
{
  auto rodDesc = std::make_unique<RodObjectDesc>("rod");
  rodDesc->restBlocks = rod.restBlocks;
  rodDesc->material = rod.material;

  runtime::ObjectRef rodRef = scene.registerObject(std::move(rodDesc));

  DERSubsystemDesc* der = findDERSubsystem(scene);
  if (der == nullptr) {
    auto created = std::make_unique<DERSubsystemDesc>();
    der = created.get();
    scene.addSubsystem(std::move(created));
  }
  der->addRod(rodRef.id);
  return rodRef;
}

void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef rod,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target)
{
  if (!rod.isA<RodObject>()) {
    throw std::runtime_error("cannot add DER constraint to non-rod object");
  }
  scene.addConstraint(rod, std::move(property), sample, stiffness,
                      std::move(target));
}

}  // namespace ksk::der
