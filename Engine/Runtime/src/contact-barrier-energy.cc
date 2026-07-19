#include <Runtime/contact-barrier-energy.h>

#include <Contact/contact-barrier.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ksk::runtime
{
    namespace
    {
        int stencilPointCount(ContactCase type)
        {
            switch (type)
            {
            case ContactCase::PP:
                return 2;
            case ContactCase::PE:
                return 3;
            case ContactCase::PT:
            case ContactCase::EE:
                return 4;
            }
            return 0;
        }

        engine::contact::EBarrierStencil barrierStencilType(ContactCase type)
        {
            switch (type)
            {
            case ContactCase::PP:
                return engine::contact::EBarrierStencil::PP;
            case ContactCase::PE:
                return engine::contact::EBarrierStencil::PE;
            case ContactCase::PT:
                return engine::contact::EBarrierStencil::PT;
            case ContactCase::EE:
                return engine::contact::EBarrierStencil::EE;
            }
            return engine::contact::EBarrierStencil::PP;
        }

        engine::contact::GIPCBarrierStencil constructBarrierStencil(
            const GlobalGeometryManager& geometry,
            const ContactStencil& contact)
        {
            engine::contact::GIPCBarrierStencil barrier;
            barrier.type = barrierStencilType(contact.type);
            barrier.dHat = contact.dHat;
            barrier.reservedDist = contact.thickness;
            barrier.kappa = contact.stiffness;

            for (int i = 0; i < stencilPointCount(contact.type); ++i)
            {
                const PointIdx point = contact.geometryIds[static_cast<size_t>(i)];
                barrier.x[static_cast<size_t>(i)] = geometry.worldPosition(point);
                barrier.restX[static_cast<size_t>(i)] = geometry.restPosition(point);
            }
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

        bool isNonZero(const glm::dvec3& value)
        {
            return value.x != 0.0 || value.y != 0.0 || value.z != 0.0;
        }

        bool isNonZero(const glm::dmat3& value)
        {
            for (int col = 0; col < 3; col++)
            {
                for (int row = 0; row < 3; row++)
                {
                    if (value[col][row] != 0.0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        std::uint64_t pointPairKey(PointIdx row, PointIdx col)
        {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
                   static_cast<std::uint32_t>(col);
        }

        std::array<glm::dvec3, 4> gatherBarrierDirection(
            const ContactStencil& contact,
            const ConstGeometryView& geometryDirection)
        {
            std::array direction{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            for (int i = 0; i < stencilPointCount(contact.type); i++)
            {
                direction[static_cast<size_t>(i)] =
                    geometryDirection.cpu()[contact.geometryIds[static_cast<size_t>(i)]];
            }
            return direction;
        }

        void scatterBarrierValues(
            const ContactStencil& contact,
            const std::array<glm::dvec3, 4>& values,
            std::vector<PointIdx>& points,
            std::vector<glm::dvec3>& out,
            std::unordered_map<PointIdx, int>& offsets)
        {
            for (int i = 0; i < stencilPointCount(contact.type); i++)
            {
                if (!isNonZero(values[i]))
                {
                    continue;
                }

                const int offset =
                    appendPoint(points,
                                out,
                                offsets,
                                contact.geometryIds[static_cast<size_t>(i)]);
                out[offset] += values[i];
            }
        }

        void scatterBarrierHessian(
            const ContactStencil& contact,
            const engine::contact::LocalBarrierHessian& hessian,
            std::vector<ContactGeometryHessianBlock>& blocks,
            std::unordered_map<std::uint64_t, int>& offsets)
        {
            const int count = stencilPointCount(contact.type);
            for (int row = 0; row < count; row++)
            {
                const PointIdx row_point = contact.geometryIds[static_cast<size_t>(row)];
                for (int col = 0; col < count; col++)
                {
                    const glm::dmat3& value = hessian.blocks[row][col];
                    if (!isNonZero(value))
                    {
                        continue;
                    }

                    const PointIdx col_point =
                        contact.geometryIds[static_cast<size_t>(col)];
                    const std::uint64_t key = pointPairKey(row_point, col_point);
                    const auto found = offsets.find(key);
                    if (found != offsets.end())
                    {
                        blocks[found->second].value += value;
                        continue;
                    }

                    const int offset = static_cast<int>(blocks.size());
                    offsets.emplace(key, offset);
                    blocks.push_back(ContactGeometryHessianBlock{
                        .row = row_point,
                        .col = col_point,
                        .value = value,
                    });
                }
            }
        }
    } // namespace

    double computeContactEnergy(const GlobalGeometryManager& geometry,
                                const ContactStencils& contacts)
    {
        double energy = 0.0;
        for (const ContactStencil& contact : contacts)
        {
            const engine::contact::GIPCBarrierStencil barrier =
                constructBarrierStencil(geometry, contact);
            energy += engine::contact::computeGIPCBarrierEnergy(barrier);
        }
        return energy;
    }

    ContactPotentialGradient computeContactGradientWrtGeometry(
        const GlobalGeometryManager& geometry,
        const ContactStencils& contacts)
    {
        std::vector<PointIdx> points;
        std::vector<glm::dvec3> values;
        std::unordered_map<PointIdx, int> offsets;

        for (const ContactStencil& contact : contacts)
        {
            const engine::contact::GIPCBarrierStencil barrier =
                constructBarrierStencil(geometry, contact);
            const engine::contact::LocalBarrierGradient localGradient =
                engine::contact::computeGIPCBarrierGradient(barrier);
            scatterBarrierValues(contact,
                                 localGradient.gradient,
                                 points,
                                 values,
                                 offsets);
        }

        return ContactPotentialGradient{
            .points = std::move(points),
            .gradient = GeometryBuffer::FromCPU(std::move(values)),
        };
    }

    ContactPotentialGradient computeContactGradient(
        const GlobalGeometryManager& geometry,
        const ContactStencils& contacts)
    {
        return computeContactGradientWrtGeometry(geometry, contacts);
    }

    ContactGeometryHessian computeContactHessianWrtGeometry(
        const GlobalGeometryManager& geometry,
        const ContactStencils& contacts)
    {
        std::vector<ContactGeometryHessianBlock> blocks;
        std::unordered_map<std::uint64_t, int> offsets;

        for (const ContactStencil& contact : contacts)
        {
            const engine::contact::GIPCBarrierStencil barrier =
                constructBarrierStencil(geometry, contact);
            const engine::contact::LocalBarrierHessian hessian =
                engine::contact::computeGIPCBarrierHessian(barrier);
            scatterBarrierHessian(contact, hessian, blocks, offsets);
        }

        return ContactGeometryHessian{
            .blocks = std::move(blocks),
        };
    }

    ContactPotentialGradient computeContactHessianProductWrtGeometry(
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
            const engine::contact::GIPCBarrierStencil barrier =
                constructBarrierStencil(geometry, stencil);
            const std::array<glm::dvec3, 4> product =
                ksk::engine::contact::computeGIPCBarrierHessianProduct(
                    barrier, gatherBarrierDirection(stencil, geometryDirection));
            scatterBarrierValues(stencil, product, points, values, offsets);
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
        return computeContactHessianProductWrtGeometry(
            geometry,
            contacts,
            geometryDirection);
    }
} // namespace ksk::runtime
