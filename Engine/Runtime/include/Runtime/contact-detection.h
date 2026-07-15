#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <RHI/buffer.h>

#include <variant>
#include <vector>

namespace ksk::runtime {

struct SubsystemContactTable {
  SubsystemId subsystem = -1;
  ContactTable contacts;
};

struct RoutedContactTables {
  ContactTable globalContacts;
  std::vector<SubsystemContactTable> internalContacts;
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
    std::variant<RoutedContactTables, DeviceRoutedContactTables>;

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
  Real thickness = 0.0;
  Real toi = 1.0;
};

[[nodiscard]] RoutedContactTables detectContactsAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

[[nodiscard]] ContactDetectionOutput detectContactTablesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config = {});

}  // namespace ksk::runtime
