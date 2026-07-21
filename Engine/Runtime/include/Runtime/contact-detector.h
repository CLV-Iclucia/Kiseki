#pragma once

#include <Runtime/contact-detection.h>
#include <Runtime/simulation-context.h>

namespace ksk::runtime {

struct CcdStepBoundResult {
  ContactCandidates sweptCandidates;
  Real stepSizeUpperBound = 1.0;
};

class ContactDetector {
 public:
  // Rebuild the active barrier constraints at the scene's current geometry.
  void rebuildActiveContacts(SimulationContext& simulation) const;

  // Use swept CCD only to bound the step length along a proposed direction.
  [[nodiscard]] CcdStepBoundResult computeCcdStepBound(
      SimulationContext& simulation,
      const DofBuffer& direction) const;

 private:
  [[nodiscard]] static ContactDetectionConfig createDetectionConfig(
      const RuntimeScene& scene,
      Real toi);

  [[nodiscard]] static ContactDetectionConfig createCurrentBarrierConfig(
      const RuntimeScene& scene);

  static void applyRoutedContacts(SimulationContext& simulation,
                                  GlobalContactRouter routedContacts);
};

}  // namespace ksk::runtime
