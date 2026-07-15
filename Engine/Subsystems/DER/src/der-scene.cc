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
  explicit DERSubsystemDesc(std::vector<runtime::ObjectId> rods)
      : rods(std::move(rods))
  {
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "der";
  }

  void build(runtime::SubsystemBuildContext& context) const override;

 private:
  std::vector<runtime::ObjectId> rods;
};

std::unique_ptr<runtime::SubsystemDesc> createDERSubsystemDesc(
    std::vector<runtime::ObjectId> rods)
{
  return std::make_unique<DERSubsystemDesc>(std::move(rods));
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
  scene.assignObjectToSubsystem("der", rodRef.id, createDERSubsystemDesc);
  return rodRef;
}

}  // namespace ksk::der
