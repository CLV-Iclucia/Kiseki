#include "../contact-detection-backends.h"

#include <stdexcept>

namespace ksk::runtime::detail {

ContactDetectionOutput detectContactsAlongDirectionGPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  (void)geometry;
  (void)config;
  if (!geometryDirection.isGPU()) {
    throw std::runtime_error(
        "GPU contact detection requires a GPU geometry direction buffer");
  }
  if (config.storage == ContactDetectionStorage::Host) {
    throw std::runtime_error(
        "GPU contact detection host readback is not implemented yet");
  }
  // GPU contract for the future backend:
  // 1. consume device-resident geometry positions/topology plus geometryDirection;
  // 2. build trajectory AABBs for points, edges, and triangles on device;
  // 3. run GPU LBVH broad phase for PT and EE candidate streams;
  // 4. run shared ACCD on candidate streams to compute a step bound/contact set;
  // 5. activate candidates into PP/PE/PT/EE stencils or a device contact table;
  // 6. route same-subsystem contacts to subsystem-owned tables and cross-
  //    subsystem/collider contacts to global handling.
  throw std::runtime_error("GPU contact detection is not implemented yet");
}

}  // namespace ksk::runtime::detail
