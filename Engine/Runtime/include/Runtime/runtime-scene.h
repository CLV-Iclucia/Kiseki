#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

#include <Runtime/contact-table.h>
#include <Runtime/dof-layout.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/solver-config.h>
#include <Runtime/types.h>

namespace ksk::runtime {

class RuntimeSimulation;
class Subsystem;
class SubsystemBuildContext;
struct RuntimeSceneDesc;

struct ObjectRef {
  ObjectId id = -1;
  ObjectTypeId type = nullptr;
  std::string tag;

  [[nodiscard]] bool isValid() const noexcept
  {
    return id >= 0;
  }

  [[nodiscard]] ObjectTypeId typeId() const noexcept
  {
    return type;
  }

  template <typename ObjectT>
  [[nodiscard]] bool isA() const noexcept
  {
    return typeId() == elementTypeId<ObjectT>();
  }
};

struct SubsystemBatch {
  std::string type;
  int firstSubsystem = 0;
  int subsystemCount = 0;
  int qOffset = 0;
  int qCount = 0;
  int geometryOffset = 0;
  int geometryCount = 0;
  int auxOffset = 0;
  int auxCount = 0;
};

struct BufferLayout {
  int qScalars = 0;
  int qdotScalars = 0;
  int geometryPoints = 0;
};

struct RuntimeScene {
  DofLayout dofs;
  std::vector<SubsystemBatch> subsystemBatches;
  GlobalGeometryManager geometry;
  ContactTable contacts;
  GlobalSolverConfig solverConfig;
  glm::dvec3 gravity{0.0, -9.81, 0.0};
  BufferLayout buffers;

  void refreshBufferLayout() noexcept;
};

enum class PropertyType
{
  Scalar,
  Vector,
};

struct PropertyDescriptor {
  std::string name;
  PropertyType type = PropertyType::Scalar;
  int componentCount = 1;
  int sampleCount = 0;
  std::string metadata;
  ObjectId ownerId = -1;
  std::string ownerTag;
};

using ScalarConstraintTarget = std::function<double(double)>;

struct SceneConstraintDesc {
  ObjectRef object;
  std::string property;
  int sample = 0;
  double stiffness = 0.0;
  ScalarConstraintTarget target;
};

struct SceneObjectDesc {
  explicit SceneObjectDesc(std::string tag = {})
      : tag(std::move(tag))
  {
  }

  virtual ~SceneObjectDesc() = default;

  [[nodiscard]] virtual ObjectTypeId typeId() const noexcept = 0;
  [[nodiscard]] virtual std::string_view typeName() const noexcept = 0;
  [[nodiscard]] virtual std::vector<PropertyDescriptor> listProperties() const
  {
    return {};
  }
  [[nodiscard]] const std::string& getTag() const noexcept { return tag; }

  void addChildObject(ObjectId child)
  {
    childObjects.push_back(child);
  }

  [[nodiscard]] const std::vector<ObjectId>& children() const noexcept
  {
    return childObjects;
  }

 private:
  std::string tag;
  std::vector<ObjectId> childObjects;
};

template <typename DescT, typename = void>
struct SceneObjectDescMarker {
  using type = void;
};

template <typename DescT>
struct SceneObjectDescMarker<DescT, std::void_t<typename DescT::ObjectType>> {
  using type = typename DescT::ObjectType;
};

class SubsystemDesc {
 public:
  virtual ~SubsystemDesc() = default;

  [[nodiscard]] virtual std::string_view typeName() const noexcept = 0;
  virtual void build(SubsystemBuildContext& context) const = 0;
};

class SubsystemBuildContext {
 public:
  explicit SubsystemBuildContext(GlobalSolverConfig solver = {},
                                 glm::dvec3 gravity =
                                     glm::dvec3(0.0, -9.81, 0.0),
                                 const RuntimeSceneDesc* scene = nullptr);
  ~SubsystemBuildContext();

  [[nodiscard]] SubsystemId nextSubsystemId() const noexcept;
  [[nodiscard]] int nextScalarOffset() const noexcept;
  [[nodiscard]] const glm::dvec3& gravity() const noexcept;
  [[nodiscard]] const RuntimeSceneDesc& sceneDesc() const;

  void addSubsystem(std::unique_ptr<Subsystem> subsystem,
                    std::string typeName);
  [[nodiscard]] RuntimeSimulation finish();

 private:
  GlobalSolverConfig solver_;
  glm::dvec3 gravity_;
  const RuntimeSceneDesc* scene_ = nullptr;
  std::vector<std::unique_ptr<Subsystem>> subsystems_;
  std::vector<std::string> subsystem_type_names_;
};

struct SceneObjectEntry {
  ObjectId id = -1;
  ObjectTypeId type = nullptr;
  std::string tag;
  std::unique_ptr<SceneObjectDesc> element;
};

struct RuntimeSceneDesc {
  GlobalSolverConfig solverConfig;
  glm::dvec3 gravity{0.0, -9.81, 0.0};
  double timeStep = 1.0 / 60.0;
  std::vector<std::unique_ptr<SubsystemDesc>> subsystems;
  std::vector<SceneObjectEntry> elements;
  std::vector<SceneConstraintDesc> constraints;

  void addSubsystem(std::unique_ptr<SubsystemDesc> subsystem);
  void addConstraint(ObjectRef object,
                     std::string property,
                     int sample,
                     double stiffness,
                     ScalarConstraintTarget target);
  ObjectRef registerObject(ObjectTypeId type, std::string tag = {});
  ObjectRef registerObject(std::unique_ptr<SceneObjectDesc> element);
  ObjectRef findObjectById(ObjectId id) const;
  const SceneObjectDesc* findObjectDesc(ObjectId id) const;
  SceneObjectDesc* findObjectDesc(ObjectId id);
  template <typename ObjectT>
  const ObjectT* findObjectDescAs(ObjectRef ref) const
  {
    static_assert(std::is_base_of_v<SceneObjectDesc, ObjectT>,
                  "findObjectDescAs<T> expects a SceneObjectDesc type");
    using Marker = typename SceneObjectDescMarker<ObjectT>::type;
    static_assert(!std::is_void_v<Marker>,
                  "SceneObjectDesc types used with findObjectDescAs must "
                  "define using ObjectType = ...");
    if (!ref.isValid()) {
      return nullptr;
    }
    if (!ref.template isA<Marker>()) {
      return nullptr;
    }
    const SceneObjectDesc* desc = findObjectDesc(ref.id);
    if (!desc) {
      return nullptr;
    }
    return static_cast<const ObjectT*>(desc);
  }
  template <typename ObjectT>
  ObjectT* findObjectDescAs(ObjectRef ref)
  {
    static_assert(std::is_base_of_v<SceneObjectDesc, ObjectT>,
                  "findObjectDescAs<T> expects a SceneObjectDesc type");
    using Marker = typename SceneObjectDescMarker<ObjectT>::type;
    static_assert(!std::is_void_v<Marker>,
                  "SceneObjectDesc types used with findObjectDescAs must "
                  "define using ObjectType = ...");
    if (!ref.isValid()) {
      return nullptr;
    }
    if (!ref.template isA<Marker>()) {
      return nullptr;
    }
    SceneObjectDesc* desc = findObjectDesc(ref.id);
    if (!desc) {
      return nullptr;
    }
    return static_cast<ObjectT*>(desc);
  }
  [[nodiscard]] std::vector<PropertyDescriptor> listProperties(ObjectRef ref) const;
  [[nodiscard]] std::vector<ObjectRef> findChildren(ObjectRef ref) const;
  std::vector<ObjectRef> findObjects(ObjectTypeId type) const;
  std::vector<ObjectRef> findObjectsByTag(const std::string& tag) const;
  template <typename ObjectT>
  std::vector<ObjectRef> findObjects() const
  {
    return findObjects(elementTypeId<ObjectT>());
  }
};

[[nodiscard]] RuntimeSimulation buildSimulation(const RuntimeSceneDesc& scene);

}  // namespace ksk::runtime
