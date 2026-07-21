#include <FEM/fem-scene.h>

#include <FEM/fem-subsystem.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::engine::fem {
namespace {

class FEMSubsystemDesc final : public runtime::SubsystemDesc {
 public:
  explicit FEMSubsystemDesc(std::vector<runtime::ObjectId> meshes)
      : meshes(std::move(meshes))
  {
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "fem";
  }

  void build(runtime::SubsystemBuildContext& context) const override;

 private:
  std::vector<runtime::ObjectId> meshes;
};

std::unique_ptr<runtime::SubsystemDesc> createFEMSubsystemDesc(
    std::vector<runtime::ObjectId> meshes)
{
  return std::make_unique<FEMSubsystemDesc>(std::move(meshes));
}

}  // namespace

void FEMSubsystemDesc::build(runtime::SubsystemBuildContext& context) const
{
  if (meshes.empty()) {
    throw std::runtime_error("FEM subsystem requires at least one mesh");
  }

  std::vector<TetMeshDesc> runtime_meshes;
  std::vector<FEMPrimitive> runtime_primitives;
  runtime_primitives.reserve(meshes.size());
  std::vector<FEMConstraintBinding> runtime_constraints;
  const runtime::RuntimeSceneDesc& scene = context.sceneDesc();

  for (int mesh_index = 0; mesh_index < static_cast<int>(meshes.size());
       ++mesh_index) {
    const runtime::ObjectId mesh_id = meshes[mesh_index];
    runtime::ObjectRef mesh_ref = scene.findObjectById(mesh_id);
    if (const TetMeshObjectDesc* mesh_desc =
            scene.findObjectDescAs<TetMeshObjectDesc>(mesh_ref)) {
      runtime_primitives.push_back(FEMTetMeshPrimitive{
          .mesh = mesh_desc->mesh,
          .offset = {},
          .runtime = {},
      });
    } else if (const ClothMeshObjectDesc* cloth_desc =
                   scene.findObjectDescAs<ClothMeshObjectDesc>(mesh_ref)) {
      runtime_primitives.push_back(FEMClothPrimitive{
          .mesh = cloth_desc->mesh,
          .offset = {},
          .runtime = {},
      });
    } else {
      throw std::runtime_error("FEM subsystem references a missing mesh");
    }

    for (const runtime::SceneConstraintDesc& constraint : scene.constraints) {
      if (constraint.object.id == mesh_id) {
        runtime_constraints.push_back(FEMConstraintBinding{
            .mesh = mesh_index,
            .constraint = constraint,
        });
      }
    }
  }

  context.addSubsystem(
      std::make_unique<FEMSubsystem>(context.nextSubsystemId(),
                                     std::move(runtime_primitives),
                                     std::move(runtime_constraints),
                                     context.nextScalarOffset(),
                                     context.gravity()),
      std::string(typeName()));
}

runtime::ObjectRef addTetMesh(runtime::RuntimeSceneDesc& scene,
                              TetMeshDesc mesh,
                              std::string tag)
{
  auto mesh_desc = std::make_unique<TetMeshObjectDesc>(std::move(tag));
  mesh_desc->mesh = std::move(mesh);

  runtime::ObjectRef mesh_ref = scene.registerObject(std::move(mesh_desc));
  scene.assignObjectToSubsystem("fem", mesh_ref.id, createFEMSubsystemDesc);
  return mesh_ref;
}

runtime::ObjectRef addClothMesh(runtime::RuntimeSceneDesc& scene,
                                ClothMeshDesc mesh,
                                std::string tag)
{
  auto mesh_desc = std::make_unique<ClothMeshObjectDesc>(std::move(tag));
  mesh_desc->mesh = std::move(mesh);

  runtime::ObjectRef mesh_ref = scene.registerObject(std::move(mesh_desc));
  scene.assignObjectToSubsystem("fem", mesh_ref.id, createFEMSubsystemDesc);
  return mesh_ref;
}

void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef mesh,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target)
{
  if (!mesh.isA<TetMeshObject>() && !mesh.isA<ClothMeshObject>()) {
    throw std::runtime_error("cannot add FEM constraint to non-FEM mesh object");
  }
  scene.addConstraint(mesh, std::move(property), sample, stiffness,
                      std::move(target));
}

}  // namespace ksk::engine::fem
