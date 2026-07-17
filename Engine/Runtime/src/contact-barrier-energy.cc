#include <Runtime/contact-barrier-energy.h>

#include <Contact/contact-barrier.h>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ksk::runtime
{
    namespace
    {
        struct StencilEvaluation
        {
            std::array<PointIdx, 4> points{-1, -1, -1, -1};
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
            int count = 0;
            double dHat = 0.0;
            double surfaceOffset = 0.0;
            double kappa = 0.0;
        };

        int stencilPointCount(ContactCase type)
        {
            return static_cast<int>(type);
        }

        engine::contact::EBarrierStencil barrierStencilType(ContactCase type)
        {
            switch (type)
            {
            case ContactCase::PP:
                return ksk::engine::contact::EBarrierStencil::PP;
            case ContactCase::PE:
                return ksk::engine::contact::EBarrierStencil::PE;
            case ContactCase::PT:
                return ksk::engine::contact::EBarrierStencil::PT;
            case ContactCase::EE:
                return ksk::engine::contact::EBarrierStencil::EE;
            }
            return engine::contact::EBarrierStencil::PP;
        }

        StencilEvaluation gatherStencil(const GlobalGeometryManager& geometry,
                                        const ContactStencil& stencil)
        {
            StencilEvaluation evaluation;
            evaluation.points = stencil.geometryIds;
            evaluation.count = stencilPointCount(stencil.type);
            evaluation.dHat = stencil.dHat;
            evaluation.surfaceOffset = stencil.thickness;
            evaluation.kappa = stencil.stiffness;

            for (int i = 0; i < evaluation.count; ++i)
            {
                const PointIdx point = evaluation.points[i];
                evaluation.x[i] = geometry.worldPosition(point);
                evaluation.restX[i] = geometry.restPosition(point);
            }
            return evaluation;
        }

        ksk::engine::contact::GIPCBarrierStencil makeBarrierStencil(
            const StencilEvaluation& stencil,
            ContactCase type)
        {
            ksk::engine::contact::GIPCBarrierStencil barrier;
            barrier.type = barrierStencilType(type);
            barrier.x = stencil.x;
            barrier.restX = stencil.restX;
            barrier.dHat = stencil.dHat;
            barrier.reservedDist = stencil.surfaceOffset;
            barrier.kappa = stencil.kappa;
            return barrier;
        }

        int appendPoint(std::vector<PointIdx>& points,
                        std::vector<glm::dvec3>& values,
                        std::unordered_map<PointIdx, int>& offsets,
                        PointIdx point)
        {
            const auto found = offsets.find(point);
            if (found != offsets.end())
            {
                return found->second;
            }

            const int offset = static_cast<int>(points.size());
            offsets.emplace(point, offset);
            points.push_back(point);
            values.push_back(glm::dvec3(0.0));
            return offset;
        }

        std::array<glm::dvec3, 4> gatherBarrierDirection(
            const StencilEvaluation& stencil,
            ConstGeometryView geometryDirection)
        {
            std::array direction{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            for (int i = 0; i < stencil.count; i++)
                direction[i] = geometryDirection.cpu()[stencil.points[i]];
            return direction;
        }

        void scatterBarrierValues(
            const StencilEvaluation& stencil,
            const std::array<glm::dvec3, 4>& values,
            std::vector<PointIdx>& points,
            std::vector<glm::dvec3>& out,
            std::unordered_map<PointIdx, int>& offsets)
        {
            for (int i = 0; i < stencil.count; i++)
            {
                const int offset =
                    appendPoint(points, out, offsets, stencil.points[static_cast<size_t>(i)]);
                out[static_cast<size_t>(offset)] += values[static_cast<size_t>(i)];
            }
        }

        bool hasBarrierValues(const StencilEvaluation& stencil,
                              const std::array<glm::dvec3, 4>& values)
        {
            for (int i = 0; i < stencil.count; i++)
            {
                const glm::dvec3& value = values[static_cast<size_t>(i)];
                if (value.x != 0.0 || value.y != 0.0 || value.z != 0.0)
                {
                    return true;
                }
            }
            return false;
        }

        double evaluateStencil(const ContactStencil& contact,
                               const GlobalGeometryManager& geometry,
                               std::vector<PointIdx>* gradient_points,
                               std::vector<glm::dvec3>* gradient_values,
                               std::unordered_map<PointIdx, int>* gradient_offsets,
                               const ConstGeometryView* geometry_direction,
                               std::vector<PointIdx>* hessian_points,
                               std::vector<glm::dvec3>* hessian_values,
                               std::unordered_map<PointIdx, int>* hessian_offsets)
        {
            const StencilEvaluation stencil = gatherStencil(geometry, contact);
            const engine::contact::GIPCBarrierStencil barrier =
                makeBarrierStencil(stencil, contact.type);

            double energy = 0.0;
            if (gradient_points != nullptr)
            {
                const engine::contact::LocalBarrierGradient gradient =
                    engine::contact::computeGIPCBarrierGradient(barrier);
                energy = gradient.energy;
                if (hasBarrierValues(stencil, gradient.gradient))
                {
                    scatterBarrierValues(stencil,
                                         gradient.gradient,
                                         *gradient_points,
                                         *gradient_values,
                                         *gradient_offsets);
                }
            }
            else
            {
                energy = ksk::engine::contact::computeGIPCBarrierEnergy(barrier);
            }

            if (geometry_direction != nullptr)
            {
                const std::array<glm::dvec3, 4> product =
                    ksk::engine::contact::computeGIPCBarrierHessianProduct(
                        barrier, gatherBarrierDirection(stencil, *geometry_direction));
                if (hasBarrierValues(stencil, product))
                {
                    scatterBarrierValues(stencil,
                                         product,
                                         *hessian_points,
                                         *hessian_values,
                                         *hessian_offsets);
                }
            }
            return energy;
        }
    } // namespace

    double computeContactEnergy(const GlobalGeometryManager& geometry,
                                const ContactStencils& contacts)
    {
        double energy = 0.0;
        for (const ContactStencil& contact : contacts)
        {
            engine::contact::GIPCBarrierStencil barrier;
            for (int i = 0; i < stencilPointCount(contact.type); ++i)
            {
                const PointIdx point = contact.geometryIds[i];
                barrier.x[i] = geometry.worldPosition(point);
                barrier.restX[i] = geometry.restPosition(point);
            }
            barrier.type = barrierStencilType(contact.type);
            barrier.dHat = contact.dHat;
            barrier.reservedDist = contact.thickness;
            barrier.kappa = contact.stiffness;
            energy += engine::contact::computeGIPCBarrierEnergy(barrier);
        }
        return energy;
    }

    ContactPotentialGradient computeContactGradient(
        const GlobalGeometryManager& geometry,
        const ContactStencils& contacts)
    {
        std::vector<PointIdx> points;
        std::vector<glm::dvec3> values;
        std::unordered_map<PointIdx, int> offsets;

        for (const ContactStencil& contact : contacts)
        {
            engine::contact::GIPCBarrierStencil barrier;
            for (int i = 0; i < stencilPointCount(contact.type); ++i)
            {
                const PointIdx point = contact.geometryIds[i];
                barrier.x[i] = geometry.worldPosition(point);
                barrier.restX[i] = geometry.restPosition(point);
            }
            barrier.type = barrierStencilType(contact.type);
            barrier.dHat = contact.dHat;
            barrier.reservedDist = contact.thickness;
            barrier.kappa = contact.stiffness;
            auto localGradient = engine::contact::computeGIPCBarrierGradient(barrier);
        }

        return ContactPotentialGradient{
            .points = std::move(points),
            .gradient = GeometryBuffer::FromCPU(std::move(values)),
        };
    }

    ContactPotentialGradient computeContactHessianProduct(
        const GlobalGeometryManager& geometry,
        const ContactStencils& contacts,
        ConstGeometryView geometryDirection)
    {
        if (!geometryDirection.isCPU())
        {
            throw std::runtime_error(
                "GIPC contact Hessian product requires a CPU geometry direction");
        }

        std::vector<PointIdx> points;
        std::vector<glm::dvec3> values;
        std::unordered_map<PointIdx, int> offsets;

        for (const ContactStencil& stencil : contacts)
        {
            evaluateStencil(stencil,
                            geometry,
                            nullptr,
                            nullptr,
                            nullptr,
                            &geometryDirection,
                            &points,
                            &values,
                            &offsets);
        }

        return ContactPotentialGradient{
            .points = std::move(points),
            .gradient = GeometryBuffer::FromCPU(std::move(values)),
        };
    }
} // namespace ksk::runtime
