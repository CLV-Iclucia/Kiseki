#pragma once

#include <Runtime/contact-detection.h>

namespace ksk::runtime::detail {

[[nodiscard]] ContactDetectionOutput detectContactsAlongDirectionCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

[[nodiscard]] ContactDetectionOutput detectContactsAlongDirectionGPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

}  // namespace ksk::runtime::detail
