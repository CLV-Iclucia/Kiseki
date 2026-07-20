#pragma once

#include <Runtime/contact-detection.h>
#include <Runtime/simulation-context.h>

namespace ksk::runtime {

struct ContactDirectionSearchResult {
  ContactCandidates candidates;
  Real stepSizeUpperBound = 1.0;
};

class ContactDetector {
 public:
  void refreshCurrentContacts(SimulationContext& simulation) const;

  void refreshFromCandidates(SimulationContext& simulation,
                             const ContactCandidates& candidates) const;

  [[nodiscard]] ContactDirectionSearchResult updateAlongDirection(
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
