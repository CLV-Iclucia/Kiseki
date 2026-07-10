#pragma once

#include <Runtime/runtime-scene.h>

namespace ksk::runtime {

class CpuRuntimeBackend {
 public:
  void prepare(const RuntimeScene& scene);

  [[nodiscard]] int scalarCount() const noexcept { return scalar_count_; }
  [[nodiscard]] int geometryPointCount() const noexcept { return geometry_point_count_; }

 private:
  int scalar_count_ = 0;
  int geometry_point_count_ = 0;
};

}  // namespace ksk::runtime
