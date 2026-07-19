#include "../contact-detection-impl.h"

#include <stdexcept>

namespace ksk::runtime::detail {

ContactDetectionOutput runCCDOnGPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  throw std::runtime_error("GPU contact detection is not implemented");
}

}  // namespace ksk::runtime::detail
