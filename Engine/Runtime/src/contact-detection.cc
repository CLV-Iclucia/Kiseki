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
    return detail::detectContactsAlongDirectionGPU(
        geometry, geometryDirection, config);
  }
  return detail::detectContactsAlongDirectionCPU(
      geometry, geometryDirection, config);
}

RoutedContactTables detectContactsAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
  ContactDetectionOutput output =
      detectContactTablesAlongDirection(geometry, geometryDirection, config);
  if (RoutedContactTables* host_tables =
          std::get_if<RoutedContactTables>(&output)) {
    return std::move(*host_tables);
  }
  throw std::runtime_error(
      "detectContactsAlongDirection requested host contacts but the selected "
      "backend produced device contact tables");
}

}  // namespace ksk::runtime
