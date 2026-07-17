#pragma once

#include <Contact/types.h>

#include <array>

#include <glm/glm.hpp>

namespace ksk::engine::contact
{
    enum class EBarrierStencil
    {
        PP,
        PE,
        PT,
        EE,
    };

    struct GIPCBarrierStencil
    {
        EBarrierStencil type = EBarrierStencil::PP;
        std::array<glm::dvec3, 4> x{
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0)
        };
        std::array<glm::dvec3, 4> restX{
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0)
        };
        Real dHat = 0.0;
        Real reservedDist = 0.0;
        Real kappa = 0.0;
    };

    struct LocalBarrierGradient
    {
        std::array<glm::dvec3, 4> gradient{
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0)
        };
    };

    [[nodiscard]] int stencilPointCount(EBarrierStencil type) noexcept;

    [[nodiscard]] Real computeGIPCBarrierEnergy(const GIPCBarrierStencil& stencil);

    [[nodiscard]] LocalBarrierGradient computeGIPCBarrierGradient(const GIPCBarrierStencil& stencil);

    [[nodiscard]] std::array<glm::dvec3, 4> computeGIPCBarrierHessianProduct(
        const GIPCBarrierStencil& stencil,
        const std::array<glm::dvec3, 4>& direction);
} // namespace ksk::engine::contact
