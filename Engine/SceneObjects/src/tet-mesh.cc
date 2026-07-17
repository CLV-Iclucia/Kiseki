#include <SceneObjects/tet-mesh.h>

namespace ksk::scene {

std::vector<runtime::PropertyDescriptor> TetMeshObjectDesc::listProperties()
    const
{
  const int vertex_count = static_cast<int>(mesh.vertices.size());
  return {
      {"x", runtime::PropertyType::Scalar, 1, vertex_count,
       "Tet mesh vertex x-position"},
      {"y", runtime::PropertyType::Scalar, 1, vertex_count,
       "Tet mesh vertex y-position"},
      {"z", runtime::PropertyType::Scalar, 1, vertex_count,
       "Tet mesh vertex z-position"},
  };
}

}  // namespace ksk::scene
