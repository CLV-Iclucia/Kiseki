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
        Real energy = 0.0;
        std::array<glm::dvec3, 4> gradient{
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0)
        };
    };

    struct LocalBarrierHessian
    {
        std::array<std::array<glm::dmat3, 4>, 4> blocks;

        LocalBarrierHessian()
        {
            for (auto& row : blocks)
            {
                for (glm::dmat3& block : row)
                {
                    block = glm::dmat3(0.0);
                }
            }
        }
    };

    inline int numPoints(EBarrierStencil type)
    {
        switch (type)
        {
        case EBarrierStencil::PP:
            return 2;
        case EBarrierStencil::PE:
            return 3;
        case EBarrierStencil::PT:
        case EBarrierStencil::EE:
            return 4;
        }
        return 0;
    }

    [[nodiscard]] int stencilPointCount(EBarrierStencil type) noexcept;

    [[nodiscard]] Real computeGIPCBarrierEnergy(const GIPCBarrierStencil& stencil);

    [[nodiscard]] LocalBarrierGradient computeGIPCBarrierGradient(const GIPCBarrierStencil& stencil);

    [[nodiscard]] LocalBarrierHessian computeGIPCBarrierHessian(const GIPCBarrierStencil& stencil);

    [[nodiscard]] std::array<glm::dvec3, 4> computeGIPCBarrierHessianProduct(
        const GIPCBarrierStencil& stencil,
        const std::array<glm::dvec3, 4>& direction);
} // namespace ksk::engine::contact
