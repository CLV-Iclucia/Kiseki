#pragma once

#include <SceneObjects/tet-mesh.h>

#include <string>

namespace ksk::engine::fem {

using TetMaterial = scene::TetMaterial;
using TetMeshDesc = scene::TetMeshDesc;
using TetMeshObject = scene::TetMeshObject;
using TetMeshObjectDesc = scene::TetMeshObjectDesc;

[[nodiscard]] runtime::ObjectRef addTetMesh(runtime::RuntimeSceneDesc& scene,
                                            TetMeshDesc mesh,
                                            std::string tag = "tet-mesh");

void addConstraint(runtime::RuntimeSceneDesc& scene,
                   runtime::ObjectRef mesh,
                   std::string property,
                   int sample,
                   double stiffness,
                   runtime::ScalarConstraintTarget target);

}  // namespace ksk::engine::fem
