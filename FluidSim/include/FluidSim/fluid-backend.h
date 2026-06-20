// ============================================================================
// include/FluidSim/fluid-backend.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <Core/properties.h>

namespace fluid {

class FluidBackend : NonCopyable {
public:
    virtual ~FluidBackend() = default;

    virtual void initialize(const FluidScene& scene) = 0;

    virtual void step(Real dt) = 0;

    virtual void readbackParticles(FluidFrame& out) = 0;

    virtual void updateCollider(const Mesh& mesh) = 0;
    virtual void updateSolverConfig(const SolverConfig& config) = 0;
};

} // namespace fluid
