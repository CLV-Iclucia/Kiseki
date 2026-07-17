#pragma once

#include <Runtime/contact-detection.h>

namespace ksk::runtime::detail {

[[nodiscard]] GlobalContactRouter runCCDOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

[[nodiscard]] ContactDetectionOutput runCCDOnGPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

}  // namespace ksk::runtime::detail
