#include <Runtime/contact-detector.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>

namespace ksk::runtime {
namespace {

GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                             int point_count)
{
  if (direction.isCPU())
    return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
}

int candidatePointCount(EContactCandidate kind)
{
  switch (kind) {
    case EContactCandidate::PointPoint:
      return 2;
    case EContactCandidate::PointEdge:
      return 3;
    case EContactCandidate::PointTriangle:
    case EContactCandidate::EdgeEdge:
      return 4;
  }
  return 0;
}

double boundsDiagonal(const GeometryBounds& bounds)
{
  const auto extent = bounds.extent();
  return std::sqrt(extent.x * extent.x +
                   extent.y * extent.y +
                   extent.z * extent.z);
}

struct ContactCandidateSanityStats {
  double directionAvg = 0.0;
  double directionMax = 0.0;
  double pointSweptDiagonalMax = 0.0;
  double edgeSweptDiagonalMax = 0.0;
  double triangleSweptDiagonalMax = 0.0;
  int selfCandidates = 0;
  int crossSubsystemCandidates = 0;
  int colliderCandidates = 0;
  int invalidCandidates = 0;
  int pointPointCandidates = 0;
  int pointEdgeCandidates = 0;
  int pointTriangleCandidates = 0;
  int edgeEdgeCandidates = 0;
};

ContactCandidateSanityStats computeContactCandidateSanityStats(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactCandidates& candidates)
{
  ContactCandidateSanityStats stats;
  const auto& directions = geometryDirection.cpu();
  for (int point = 0; point < geometry.pointCount(); ++point) {
    const double norm = glm::length(directions[point]);
    stats.directionAvg += norm;
    stats.directionMax = std::max(stats.directionMax, norm);
    stats.pointSweptDiagonalMax =
        std::max(stats.pointSweptDiagonalMax,
                 boundsDiagonal(geometry.trajectoryPointBounds(
                     PointIdx{point}, directions, 1.0)));
  }
  if (geometry.pointCount() > 0) {
    stats.directionAvg /= static_cast<double>(geometry.pointCount());
  }

  for (int edge = 0; edge < geometry.edgeCount(); ++edge) {
    stats.edgeSweptDiagonalMax =
        std::max(stats.edgeSweptDiagonalMax,
                 boundsDiagonal(geometry.trajectoryEdgeBounds(
                     edge, directions, 1.0,
                     geometry.edges[static_cast<size_t>(edge)].radius)));
  }
  for (int triangle = 0; triangle < geometry.triangleCount(); ++triangle) {
    stats.triangleSweptDiagonalMax =
        std::max(stats.triangleSweptDiagonalMax,
                 boundsDiagonal(geometry.trajectoryTriangleBounds(
                     triangle, directions, 1.0,
                     geometry.triangles[static_cast<size_t>(triangle)].thickness)));
  }

  for (const ContactCandidate& candidate : candidates) {
    switch (candidate.kind) {
      case EContactCandidate::PointPoint:
        ++stats.pointPointCandidates;
        break;
      case EContactCandidate::PointEdge:
        ++stats.pointEdgeCandidates;
        break;
      case EContactCandidate::PointTriangle:
        ++stats.pointTriangleCandidates;
        break;
      case EContactCandidate::EdgeEdge:
        ++stats.edgeEdgeCandidates;
        break;
    }

    const int point_count = candidatePointCount(candidate.kind);
    const GeometryStencilInfo info =
        geometry.classify(std::span(candidate.geometryIds.data(), point_count));
    if (!info.valid) {
      ++stats.invalidCandidates;
    }
    if (info.hasCollider) {
      ++stats.colliderCandidates;
    }
    if (info.crossesSubsystems) {
      ++stats.crossSubsystemCandidates;
    } else if (!info.hasCollider && info.subsystemCount == 1) {
      ++stats.selfCandidates;
    }
  }
  return stats;
}

void warnIfContactCandidatesLookSuspicious(
    const RuntimeScene& scene,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& currentConfig,
    const ContactCandidateDetectionResult& detected)
{
  const GlobalGeometryManager& geometry = scene.geometry;
  const int primitive_count =
      geometry.pointCount() + geometry.edgeCount() + geometry.triangleCount();
  const size_t candidate_count = detected.candidates.size();
  const size_t warning_threshold =
      std::max<size_t>(100000, static_cast<size_t>(primitive_count) * 50);
  if (candidate_count <= warning_threshold) {
    return;
  }

  const ContactCandidateSanityStats stats =
      computeContactCandidateSanityStats(
          geometry, geometryDirection, detected.candidates);

  GeometryBuffer zero_direction = GeometryBuffer::CPU(geometry.pointCount());
  const ContactCandidates zero_candidates =
      detectContactCandidatesAlongDirection(
          geometry,
          zero_direction,
          currentConfig);

  std::cerr << "[contact-sanity] warning: unusually many CCD candidates: "
            << candidate_count << " primitives=" << primitive_count
            << " zero_direction_candidates=" << zero_candidates.size()
            << " alpha_bound=" << detected.stepSizeUpperBound << '\n';
  std::cerr << "[contact-sanity] candidate types point_point="
            << stats.pointPointCandidates
            << " point_edge=" << stats.pointEdgeCandidates
            << " point_triangle=" << stats.pointTriangleCandidates
            << " edge_edge=" << stats.edgeEdgeCandidates
            << " self=" << stats.selfCandidates
            << " cross_subsystem=" << stats.crossSubsystemCandidates
            << " collider=" << stats.colliderCandidates
            << " invalid=" << stats.invalidCandidates << '\n';
  std::cerr << "[contact-sanity] direction norm avg=" << stats.directionAvg
            << " max=" << stats.directionMax
            << " swept_diag_max point=" << stats.pointSweptDiagonalMax
            << " edge=" << stats.edgeSweptDiagonalMax
            << " triangle=" << stats.triangleSweptDiagonalMax << '\n';
}

}  // namespace

ContactDetectionConfig ContactDetector::createDetectionConfig(
    const RuntimeScene& scene,
    Real toi)
{
  const GlobalSolverConfig& solver_config = scene.solverConfig;
  const Real barrier_distance =
      solver_config.contactBarrierDistance > 0.0
          ? solver_config.contactBarrierDistance
          : solver_config.contactDetectionDistance;
  return ContactDetectionConfig{
      .storage = solver_config.contactDetectionStorage,
      .detectionDistance = solver_config.contactDetectionDistance,
      .dHat = barrier_distance,
      .stiffness = solver_config.contactStiffness,
      .toi = toi,
  };
}

ContactDetectionConfig ContactDetector::createCurrentBarrierConfig(
    const RuntimeScene& scene)
{
  ContactDetectionConfig config = createDetectionConfig(scene, 0.0);
  config.detectionDistance = config.dHat;
  return config;
}

void ContactDetector::applyRoutedContacts(
    SimulationContext& simulation,
    GlobalContactRouter routedContacts)
{
  RuntimeScene& scene = simulation.scene();
  scene.contacts = std::move(routedContacts.globalContacts);

  for (const auto& subsystem : simulation.subsystems()) {
    ContactStencils contacts;
    const auto found =
        routedContacts.subsystemInternalContacts.find(subsystem->id());
    if (found != routedContacts.subsystemInternalContacts.end()) {
      contacts = std::move(found->second.contacts);
    }
    subsystem->applyInternalContacts(std::move(contacts));
  }
}

void ContactDetector::rebuildActiveContacts(
    SimulationContext& simulation) const
{
  GeometryBuffer zero_direction =
      GeometryBuffer::CPU(simulation.scene().geometry.pointCount());
  GlobalContactRouter routed_contacts =
      runCCD(simulation.scene().geometry,
             zero_direction,
             createCurrentBarrierConfig(simulation.scene()));
  applyRoutedContacts(simulation, std::move(routed_contacts));
}

CcdStepBoundResult ContactDetector::computeCcdStepBound(
    SimulationContext& simulation,
    const DofBuffer& direction) const
{
  RuntimeScene& scene = simulation.scene();
  GeometryBuffer geometry_direction =
      createGeometryDirectionBuffer(direction, scene.geometry.pointCount());

  for (const auto& subsystem : simulation.subsystems()) {
    subsystem->mapLocalDirectionToGeometry(
        direction.slice(subsystem->dofRange()), geometry_direction.view());
  }

  const ContactDetectionConfig ccd_config = createDetectionConfig(scene, 1.0);
  const ContactDetectionConfig current_config =
      createCurrentBarrierConfig(scene);
  ContactCandidateDetectionResult detected =
      detectContactCandidatesAndStepSizeAlongDirection(
          scene.geometry,
          geometry_direction,
          ccd_config);
  warnIfContactCandidatesLookSuspicious(
      scene, geometry_direction, current_config, detected);
  return CcdStepBoundResult{
      .sweptCandidates = std::move(detected.candidates),
      .stepSizeUpperBound = detected.stepSizeUpperBound,
  };
}

}  // namespace ksk::runtime
