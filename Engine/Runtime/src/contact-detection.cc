#include <Runtime/contact-detection.h>

#include "contact-detection-backends.h"

#include <stdexcept>
#include <utility>
#include <variant>

namespace ksk::runtime {

ContactDetectionOutput detectContactTablesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  if (geometryDirection.isGPU()) {
    return detail::runCCDOnGPU(
        geometry, geometryDirection, config);
  }
  return detail::runCCDOnCPU(geometry, geometryDirection, config);
}

GlobalContactRouter runCCD(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error("runCCD requested host tables but got device tables");
    }
    return detail::runCCDOnCPU(geometry, geometryDirection, config);
}

}  // namespace ksk::runtime
