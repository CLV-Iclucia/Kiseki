#include <Runtime/contact-detector.h>

#include <utility>

namespace ksk::runtime {
namespace {

GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                             int point_count)
{
  if (direction.isCPU())
    return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
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

void ContactDetector::refreshCurrentContacts(
    SimulationContext& simulation) const
{
  GeometryBuffer zero_direction =
      GeometryBuffer::CPU(simulation.scene().geometry.pointCount());
  const ContactCandidates candidates = detectContactCandidatesAlongDirection(
      simulation.scene().geometry,
      zero_direction,
      createCurrentBarrierConfig(simulation.scene()));
  refreshFromCandidates(simulation, candidates);
}

void ContactDetector::refreshFromCandidates(
    SimulationContext& simulation,
    const ContactCandidates& candidates) const
{
  GlobalContactRouter routed_contacts =
      refreshActiveContactsFromCandidates(
          simulation.scene().geometry,
          candidates,
          createCurrentBarrierConfig(simulation.scene()));
  applyRoutedContacts(simulation, std::move(routed_contacts));
}

ContactDirectionSearchResult ContactDetector::updateAlongDirection(
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

  ContactCandidateDetectionResult detected =
      detectContactCandidatesAndStepSizeAlongDirection(
          scene.geometry,
          geometry_direction,
          createDetectionConfig(scene, 1.0));
  refreshFromCandidates(simulation, detected.candidates);
  return ContactDirectionSearchResult{
      .candidates = std::move(detected.candidates),
      .stepSizeUpperBound = detected.stepSizeUpperBound,
  };
}

}  // namespace ksk::runtime
