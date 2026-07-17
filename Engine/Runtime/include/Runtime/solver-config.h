#pragma once

#include <Runtime/contact-detection.h>

namespace ksk::runtime {

struct GlobalSolverConfig {
  int maxNewtonIterations = 25;
  int maxPcgIterations = 200;
  int maxLineSearchIterations = 12;
  double newtonGradientTolerance = 1.0e-8;
  double newtonStepTolerance = 1.0e-10;
  double lineSearchArmijo = 1.0e-4;
  double lineSearchShrink = 0.5;
  double pcgTolerance = 1.0e-8;
  ContactDetectionStorage contactDetectionStorage =
      ContactDetectionStorage::Host;
  double contactDetectionDistance = 0.0;
  double contactBarrierDistance = 0.0;
  double contactStiffness = 0.0;
  double contactThickness = 0.0;
};

}  // namespace ksk::runtime
