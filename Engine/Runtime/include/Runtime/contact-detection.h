#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <RHI/buffer.h>

#include <array>
#include <variant>
#include <vector>
#include <map>

namespace ksk::runtime {

struct SubsystemContactData {
  SubsystemId subsystem = -1;
  ContactStencils contacts;
};

struct GlobalContactRouter {
  ContactStencils globalContacts;
  std::map<SubsystemId, SubsystemContactData> subsystemInternalContacts;
};

enum class EContactCandidate : std::uint16_t {
  PointTriangle,
  EdgeEdge,
  PointPoint,
  PointEdge,
};

struct ContactCandidate {
  EContactCandidate kind = EContactCandidate::PointPoint;
  std::array<int, 4> geometryIds{-1, -1, -1, -1};
  Real detectionDistance = 0.0;
  Real dHat = 0.0;
  Real stiffness = 0.0;
  Real reservedDistance = 0.0;
};

using ContactCandidates = std::vector<ContactCandidate>;

struct ContactWorkList {
  std::vector<std::array<int, 2>> deformablePointTriangles;
  std::vector<std::array<int, 2>> deformableEdgeEdges;

  [[nodiscard]] bool empty() const noexcept
  {
    return deformablePointTriangles.empty() && deformableEdgeEdges.empty();
  }
};

struct DeviceContactTable {
  rhi::BufferRef stencils;
  rhi::BufferRef batches;
  rhi::BufferRef counts;
};

struct SubsystemDeviceContactTable {
  SubsystemId subsystem = -1;
  DeviceContactTable contacts;
};

struct DeviceRoutedContactTables {
  DeviceContactTable globalContacts;
  std::vector<SubsystemDeviceContactTable> internalContacts;
};

using ContactDetectionOutput =
    std::variant<GlobalContactRouter, DeviceRoutedContactTables>;

struct ContactCandidateDetectionResult {
  ContactCandidates candidates;
  Real stepSizeUpperBound = 1.0;
};

enum class ContactDetectionStorage {
  Auto,
  Host,
  Device,
};

struct ContactDetectionConfig {
  ContactDetectionStorage storage = ContactDetectionStorage::Auto;
  Real detectionDistance = 1.0e-3;
  Real dHat = 1.0e-3;
  Real stiffness = 1.0e5;
  Real toi = 1.0;
};

[[nodiscard]] GlobalContactRouter runCCD(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

[[nodiscard]] ContactCandidates detectContactCandidatesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

[[nodiscard]] ContactWorkList gatherCollisionWorkListAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

[[nodiscard]] ContactCandidateDetectionResult
detectContactCandidatesAndStepSizeAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

[[nodiscard]] GlobalContactRouter refreshActiveContactsFromCandidates(
    const GlobalGeometryManager& geometry,
    const ContactCandidates& candidates,
    const ContactDetectionConfig& config = {});

[[nodiscard]] ContactDetectionOutput detectContactTablesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

}  // namespace ksk::runtime
