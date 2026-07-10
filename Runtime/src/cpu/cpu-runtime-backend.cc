#include <Runtime/cpu/cpu-runtime-backend.h>

namespace ksk::runtime {

void CpuRuntimeBackend::prepare(const RuntimeScene& scene)
{
  scalar_count_ = scene.dofs.totalScalars;
  geometry_point_count_ = static_cast<int>(scene.geometry.points.size());
}

}  // namespace ksk::runtime
