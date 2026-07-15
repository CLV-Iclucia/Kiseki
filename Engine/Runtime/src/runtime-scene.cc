#include <Runtime/runtime-scene.h>

#include <Runtime/runtime-simulation.h>
#include <Runtime/subsystem.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ksk::runtime {

void RuntimeScene::refreshBufferLayout() noexcept
{
  buffers.qScalars = dofs.totalScalars;
  buffers.qdotScalars = dofs.totalScalars;
  buffers.geometryPoints = geometry.pointCount();
}

SubsystemBuildContext::SubsystemBuildContext(GlobalSolverConfig solver,
                                             glm::dvec3 gravity,
                                             const RuntimeSceneDesc* scene)
    : solver_(solver)
    , gravity_(gravity)
    , scene_(scene)
{
}

SubsystemBuildContext::~SubsystemBuildContext() = default;

SubsystemId SubsystemBuildContext::nextSubsystemId() const noexcept
{
  return SubsystemId{static_cast<int>(subsystems_.size())};
}

int SubsystemBuildContext::nextScalarOffset() const noexcept
{
  int scalar_offset = 0;
  for (const auto& subsystem : subsystems_) {
    const DofRange range = subsystem->dofRange();
    scalar_offset =
        std::max(scalar_offset, range.scalarOffset + range.scalarCount);
  }
  return scalar_offset;
}

const glm::dvec3& SubsystemBuildContext::gravity() const noexcept
{
  return gravity_;
}

const RuntimeSceneDesc& SubsystemBuildContext::sceneDesc() const
{
  if (!scene_) {
    throw std::runtime_error("subsystem build context has no scene description");
  }
  return *scene_;
}

void SubsystemBuildContext::addSubsystem(std::unique_ptr<Subsystem> subsystem,
                                         std::string typeName)
{
  if (!subsystem) {
    throw std::runtime_error("cannot add null subsystem");
  }
  if (typeName.empty()) {
    throw std::runtime_error("cannot add subsystem with empty type name");
  }
  subsystems_.push_back(std::move(subsystem));
  subsystem_type_names_.push_back(std::move(typeName));
}

RuntimeSimulation SubsystemBuildContext::finish()
{
  return RuntimeSimulation(
      std::move(subsystems_),
      std::move(subsystem_type_names_),
      solver_,
      gravity_);
}

void RuntimeSceneDesc::addSubsystem(std::unique_ptr<SubsystemDesc> subsystem)
{
  if (!subsystem) {
    throw std::runtime_error("cannot add null subsystem description");
  }
  subsystems.push_back(std::move(subsystem));
}

void RuntimeSceneDesc::assignObjectToSubsystem(
    std::string subsystemType,
    ObjectId object,
    SubsystemDescFactory factory)
{
  if (subsystemType.empty()) {
    throw std::runtime_error("cannot assign object to an empty subsystem type");
  }
  if (!findObjectById(object).isValid()) {
    throw std::runtime_error("cannot assign invalid object to subsystem");
  }
  if (!factory) {
    throw std::runtime_error("cannot assign object with an empty subsystem factory");
  }

  for (SubsystemObjectGroup& group : subsystemObjectGroups) {
    if (group.type == subsystemType) {
      group.objects.push_back(object);
      return;
    }
  }

  subsystemObjectGroups.push_back(SubsystemObjectGroup{
      .type = std::move(subsystemType),
      .objects = {object},
      .factory = std::move(factory),
  });
}

void RuntimeSceneDesc::addConstraint(ObjectRef object,
                                     std::string property,
                                     int sample,
                                     double stiffness,
                                     ScalarConstraintTarget target)
{
  if (!object.isValid() || !findObjectById(object.id).isValid()) {
    throw std::runtime_error("cannot add constraint to an invalid object");
  }
  if (property.empty()) {
    throw std::runtime_error("cannot add constraint with an empty property");
  }
  if (!target) {
    throw std::runtime_error("cannot add constraint with an empty target");
  }
  constraints.push_back(SceneConstraintDesc{
      .object = std::move(object),
      .property = std::move(property),
      .sample = sample,
      .stiffness = stiffness,
      .target = std::move(target),
  });
}

ObjectRef RuntimeSceneDesc::registerObject(ObjectTypeId type,
                                              std::string tag)
{
  const ObjectId id{static_cast<int>(elements.size())};
  elements.push_back(SceneObjectEntry{id, type, std::move(tag), nullptr});
  return ObjectRef{id, type, elements.back().tag};
}

ObjectRef RuntimeSceneDesc::registerObject(std::unique_ptr<SceneObjectDesc> element)
{
  if (!element) {
    throw std::runtime_error("cannot register null scene element");
  }
  const ObjectId id{static_cast<int>(elements.size())};
  const ObjectTypeId type = element->typeId();
  const std::string tag = element->getTag();
  elements.push_back(
      SceneObjectEntry{id, type, tag, std::move(element)});
  return ObjectRef{id, type, elements.back().tag};
}

const SceneObjectDesc* RuntimeSceneDesc::findObjectDesc(ObjectId id) const
{
  if (id < 0 || id >= static_cast<int>(elements.size())) {
    return nullptr;
  }
  const SceneObjectEntry& entry = elements[static_cast<size_t>(id)];
  if (entry.id != id) {
    return nullptr;
  }
  return entry.element.get();
}

SceneObjectDesc* RuntimeSceneDesc::findObjectDesc(ObjectId id)
{
  if (id < 0 || id >= static_cast<int>(elements.size())) {
    return nullptr;
  }
  SceneObjectEntry& entry = elements[static_cast<size_t>(id)];
  if (entry.id != id) {
    return nullptr;
  }
  return entry.element.get();
}

std::vector<PropertyDescriptor> RuntimeSceneDesc::listProperties(ObjectRef ref) const
{
  const SceneObjectDesc* desc = findObjectDesc(ref.id);
  if (!desc) {
    return {};
  }
  auto properties = desc->listProperties();
  for (auto& property : properties) {
    property.ownerId = ref.id;
    property.ownerTag = ref.tag;
  }
  return properties;
}

std::vector<ObjectRef> RuntimeSceneDesc::findChildren(ObjectRef ref) const
{
  std::vector<ObjectRef> result;
  const SceneObjectDesc* desc = findObjectDesc(ref.id);
  if (!desc) {
    return result;
  }
  for (ObjectId childId : desc->children()) {
    result.push_back(findObjectById(childId));
  }
  return result;
}

ObjectRef RuntimeSceneDesc::findObjectById(ObjectId id) const
{
  if (id < 0 || id >= static_cast<int>(elements.size())) {
    return ObjectRef{};
  }
  const SceneObjectEntry& entry = elements[static_cast<size_t>(id)];
  if (entry.id != id) {
    return ObjectRef{};
  }
  return ObjectRef{entry.id, entry.type, entry.tag};
}

std::vector<ObjectRef> RuntimeSceneDesc::findObjects(ObjectTypeId type) const
{
  std::vector<ObjectRef> result;
  result.reserve(elements.size());
  for (const auto& entry : elements) {
    if (entry.type == type) {
      result.emplace_back(ObjectRef{entry.id, entry.type, entry.tag});
    }
  }
  return result;
}

std::vector<ObjectRef> RuntimeSceneDesc::findObjectsByTag(const std::string& tag) const
{
  std::vector<ObjectRef> result;
  result.reserve(elements.size());
  for (const auto& entry : elements) {
    if (entry.tag == tag) {
      result.emplace_back(ObjectRef{entry.id, entry.type, entry.tag});
    }
  }
  return result;
}

RuntimeSimulation buildSimulation(const RuntimeSceneDesc& scene)
{
  SubsystemBuildContext context(scene.solverConfig, scene.gravity, &scene);
  for (const auto& subsystem : scene.subsystems) {
    if (!subsystem) {
      throw std::runtime_error("runtime scene contains null subsystem");
    }
    subsystem->build(context);
  }
  for (const SubsystemObjectGroup& group : scene.subsystemObjectGroups) {
    if (!group.factory) {
      throw std::runtime_error("runtime scene contains empty subsystem factory");
    }
    std::unique_ptr<SubsystemDesc> subsystem = group.factory(group.objects);
    if (!subsystem) {
      throw std::runtime_error(
          "runtime scene subsystem factory produced null subsystem");
    }
    subsystem->build(context);
  }
  return context.finish();
}

}  // namespace ksk::runtime
