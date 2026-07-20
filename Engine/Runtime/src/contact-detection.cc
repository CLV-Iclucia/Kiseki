#include <Runtime/contact-detection.h>

#include "contact-detection-impl.h"

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

ContactCandidates detectContactCandidatesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error(
            "detectContactCandidatesAlongDirection requested host tables but got device tables");
    }
    return detail::detectContactCandidatesAlongDirectionOnCPU(
        geometry, geometryDirection, config);
}

ContactWorkList gatherCollisionWorkListAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error(
            "gatherCollisionWorkListAlongDirection requested host work lists but got device geometry");
    }
    return detail::gatherCollisionWorkListAlongDirectionOnCPU(
        geometry, geometryDirection, config);
}

ContactCandidateDetectionResult
detectContactCandidatesAndStepSizeAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error(
            "detectContactCandidatesAndStepSizeAlongDirection requested host tables but got device tables");
    }
    return detail::detectContactCandidatesAndStepSizeAlongDirectionOnCPU(
        geometry, geometryDirection, config);
}

GlobalContactRouter refreshActiveContactsFromCandidates(
    const GlobalGeometryManager& geometry,
    const ContactCandidates& candidates,
    const ContactDetectionConfig& config)
{
    return detail::refreshActiveContactsFromCandidatesOnCPU(
        geometry, candidates, config);
}

}  // namespace ksk::runtime
