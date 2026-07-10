#include <Runtime/runtime-scene.h>

namespace ksk::runtime {

void RuntimeScene::refreshBufferLayout() noexcept
{
  buffers.qScalars = dofs.totalScalars;
  buffers.qdotScalars = dofs.totalScalars;
  buffers.geometryPoints = static_cast<int>(geometry.points.size());
}

}  // namespace ksk::runtime
