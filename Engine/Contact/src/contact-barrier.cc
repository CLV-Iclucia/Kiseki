#include <Contact/contact-barrier.h>

#include <Contact/gipc/barrier.h>
#include <Contact/gipc/pfpx.h>
#include <Contact/shared/gipc-mollifier.h>
#include <Contact/shared/ipc-distance.h>

#include <Eigen/Dense>

#include <algorithm>

namespace ksk::engine::contact
{
    namespace
    {
        Real effectiveI5(Real distance_sqr, Real reserved_dist, Real d_hat)
        {
            const Real distance = std::sqrt(std::max(distance_sqr, 0.0));
            const Real gap = std::max(distance - reserved_dist, d_hat * 1.0e-12);
            return (gap * gap) / (d_hat * d_hat);
        }

        Real stencilI5(const GIPCBarrierStencil& stencil)
        {
            using namespace ksk::engine::contact::shared;
            switch (stencil.type)
            {
            case EBarrierStencil::PP:
                return effectiveI5(shDistanceSqrPointPoint(stencil.x[0], stencil.x[1]),
                                   stencil.reservedDist,
                                   stencil.dHat);
            case EBarrierStencil::PE:
                return effectiveI5(
                    shDistanceSqrPointLine(stencil.x[0], stencil.x[1], stencil.x[2]),
                    stencil.reservedDist,
                    stencil.dHat);
            case EBarrierStencil::PT:
                return effectiveI5(shDistanceSqrPointPlane(stencil.x[0],
                                                           stencil.x[1],
                                                           stencil.x[2],
                                                           stencil.x[3]),
                                   stencil.reservedDist,
                                   stencil.dHat);
            case EBarrierStencil::EE:
                return effectiveI5(shDistanceSqrLineLine(stencil.x[0],
                                                         stencil.x[1],
                                                         stencil.x[2],
                                                         stencil.x[3]),
                                   stencil.reservedDist,
                                   stencil.dHat);
            }
            return 1.0;
        }

        template <int N>
        void scatterFlat(std::array<glm::dvec3, 4>& out,
                         const Eigen::Matrix<Real, N * 3, 1>& flat)
        {
            for (int i = 0; i < N; i++)
                out[i] += glm::dvec3(flat[3 * i + 0], flat[3 * i + 1], flat[3 * i + 2]);
        }

        template <int N>
        Eigen::Matrix<Real, N * 3, 1> flattenDirection(
            const std::array<glm::dvec3, 4>& direction)
        {
            Eigen::Matrix<Real, N * 3, 1> flat;
            for (int i = 0; i < N; i++)
            {
                flat[3 * i + 0] = direction[i].x;
                flat[3 * i + 1] = direction[i].y;
                flat[3 * i + 2] = direction[i].z;
            }
            return flat;
        }

        template <int N, int VecDim>
        void appendRankOneGradient(
            const GIPCBarrierStencil& stencil,
            const Eigen::Matrix<Real, VecDim, N * 3>& pfpx,
            const Eigen::Matrix<Real, VecDim, 1>& q0,
            Real i5,
            LocalBarrierGradient& result)
        {
            if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0 || i5 <= 0.0 ||
                i5 >= 1.0)
            {
                return;
            }

            const gipc::Barrier barrier(stencil.dHat);
            const Real sqrt_i5 = std::sqrt(i5);
            const Eigen::Matrix<Real, VecDim, 1> pk1 =
                q0 * (barrier.gradCoeff(i5) * sqrt_i5);
            const Eigen::Matrix<Real, N * 3, 1> flat_gradient =
                stencil.kappa * pfpx.transpose() * pk1;
            scatterFlat<N>(result.gradient, flat_gradient);
        }

        template <int N, int VecDim>
        std::array<glm::dvec3, 4> rankOneHessianProduct(
            const GIPCBarrierStencil& stencil,
            const Eigen::Matrix<Real, VecDim, N * 3>& pfpx,
            const Eigen::Matrix<Real, VecDim, 1>& q0,
            Real i5,
            const std::array<glm::dvec3, 4>& direction)
        {
            std::array result{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0)};
            if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0 || i5 <= 0.0 ||
                i5 >= 1.0)
            {
                return result;
            }

            const gipc::Barrier barrier(stencil.dHat);
            const Real lambda = stencil.kappa * barrier.clampedLambda0(i5);
            if (lambda <= 0.0)
            {
                return result;
            }
            const Eigen::Matrix<Real, N * 3, 1> v = pfpx.transpose() * q0;
            const Eigen::Matrix<Real, N * 3, 1> dx = flattenDirection<N>(direction);
            scatterFlat<N>(result, lambda * v * v.dot(dx));
            return result;
        }

        Real addMollifiedEEGradient(const GIPCBarrierStencil& stencil,
                                       LocalBarrierGradient& result)
        {
            const auto mollified = shared::shMollifiedBarrierEEWithOffset(
                stencil.x[0],
                stencil.x[1],
                stencil.x[2],
                stencil.x[3],
                stencil.restX[0],
                stencil.restX[1],
                stencil.restX[2],
                stencil.restX[3],
                stencil.dHat,
                stencil.kappa,
                stencil.reservedDist);
            if (!mollified.active)
            {
                return 0.0;
            }

            Eigen::Matrix<Real, 12, 1> flat_gradient;
            for (int i = 0; i < 12; i++)
            {
                flat_gradient[i] = mollified.grad[i];
            }
            scatterFlat<4>(result.gradient, flat_gradient);

            return shared::shMollifiedEnergyEEWithOffset(stencil.x[0],
                                                         stencil.x[1],
                                                         stencil.x[2],
                                                         stencil.x[3],
                                                         stencil.restX[0],
                                                         stencil.restX[1],
                                                         stencil.restX[2],
                                                         stencil.restX[3],
                                                         stencil.dHat,
                                                         stencil.kappa,
                                                         stencil.reservedDist);
        }

        std::array<glm::dvec3, 4> mollifiedEEHessianProduct(
            const GIPCBarrierStencil& stencil,
            const std::array<glm::dvec3, 4>& direction)
        {
            std::array<glm::dvec3, 4> result{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            const auto mollified = shared::shMollifiedBarrierEEWithOffset(
                stencil.x[0],
                stencil.x[1],
                stencil.x[2],
                stencil.x[3],
                stencil.restX[0],
                stencil.restX[1],
                stencil.restX[2],
                stencil.restX[3],
                stencil.dHat,
                stencil.kappa,
                stencil.reservedDist);
            if (!mollified.active)
            {
                return result;
            }

            const Eigen::Matrix<Real, 12, 1> dx = flattenDirection<4>(direction);
            Eigen::Matrix<Real, 12, 1> flat_product = Eigen::Matrix<Real, 12, 1>::Zero();
            for (int rank = 0; rank < mollified.rank; rank++)
            {
                Eigen::Matrix<Real, 12, 1> w;
                for (int i = 0; i < 12; i++)
                {
                    w[i] = mollified.w[rank][i];
                }
                flat_product += mollified.lam[rank] * w * w.dot(dx);
            }
            scatterFlat<4>(result, flat_product);
            return result;
        }
    } // namespace

    Real computeGIPCBarrierEnergy(const GIPCBarrierStencil& stencil)
    {
        if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0)
        {
            return 0.0;
        }
        gipc::Barrier barrier(stencil.dHat);
        Real i5 = stencilI5(stencil);
        return stencil.kappa * barrier.energy(i5 * stencil.dHat * stencil.dHat);
    }

    LocalBarrierGradient computeGIPCBarrierGradient(const GIPCBarrierStencil& stencil)
    {
        LocalBarrierGradient result;
        if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0)
        {
            return result;
        }

        switch (stencil.type)
        {
        case EBarrierStencil::PP:
            {
                const auto pfpx =
                    gipc::computePFPx_PP(stencil.x[0], stencil.x[1], stencil.dHat, stencil.reservedDist);
                if (pfpx.valid)
                    appendRankOneGradient<2, 3>(stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), result);
                break;
            }
        case EBarrierStencil::PE:
            {
                const auto pfpx = gipc::computePFPx_PE(
                    stencil.x[0], stencil.x[1], stencil.x[2], stencil.dHat);
                if (pfpx.valid)
                    appendRankOneGradient<3, 6>(stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), result);
                break;
            }
        case EBarrierStencil::PT:
            {
                const auto pfpx = gipc::computePFPx_PT(stencil.x[0],
                                                       stencil.x[1],
                                                       stencil.x[2],
                                                       stencil.x[3],
                                                       stencil.dHat);
                if (pfpx.valid)
                    appendRankOneGradient<4, 9>(stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), result);
                break;
            }
        case EBarrierStencil::EE:
            {
                if (shared::shEEUsesMollifier(stencil.x[0],
                                              stencil.x[1],
                                              stencil.x[2],
                                              stencil.x[3],
                                              stencil.restX[0],
                                              stencil.restX[1],
                                              stencil.restX[2],
                                              stencil.restX[3]))
                {
                    addMollifiedEEGradient(stencil, result);
                    break;
                }
                const auto pfpx = gipc::computePFPx_EE(stencil.x[0],
                                                       stencil.x[1],
                                                       stencil.x[2],
                                                       stencil.x[3],
                                                       stencil.dHat);
                if (pfpx.valid)
                {
                    appendRankOneGradient<4, 9>(
                        stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), result);
                }
                break;
            }
        }
        return result;
    }

    std::array<glm::dvec3, 4> computeGIPCBarrierHessianProduct(
        const GIPCBarrierStencil& stencil,
        const std::array<glm::dvec3, 4>& direction)
    {
        if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0)
        {
            return {
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
        }

        switch (stencil.type)
        {
        case EBarrierStencil::PP:
            {
                const auto pfpx =
                    gipc::computePFPx_PP(stencil.x[0], stencil.x[1], stencil.dHat);
                if (pfpx.valid)
                {
                    return rankOneHessianProduct<2, 3>(
                        stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), direction);
                }
                break;
            }
        case EBarrierStencil::PE:
            {
                const auto pfpx = gipc::computePFPx_PE(
                    stencil.x[0], stencil.x[1], stencil.x[2], stencil.dHat);
                if (pfpx.valid)
                {
                    return rankOneHessianProduct<3, 6>(
                        stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), direction);
                }
                break;
            }
        case EBarrierStencil::PT:
            {
                const auto pfpx = gipc::computePFPx_PT(stencil.x[0],
                                                       stencil.x[1],
                                                       stencil.x[2],
                                                       stencil.x[3],
                                                       stencil.dHat);
                if (pfpx.valid)
                {
                    return rankOneHessianProduct<4, 9>(
                        stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), direction);
                }
                break;
            }
        case EBarrierStencil::EE:
            {
                if (shared::shEEUsesMollifier(stencil.x[0],
                                              stencil.x[1],
                                              stencil.x[2],
                                              stencil.x[3],
                                              stencil.restX[0],
                                              stencil.restX[1],
                                              stencil.restX[2],
                                              stencil.restX[3]))
                {
                    return mollifiedEEHessianProduct(stencil, direction);
                }
                const auto pfpx = gipc::computePFPx_EE(stencil.x[0],
                                                       stencil.x[1],
                                                       stencil.x[2],
                                                       stencil.x[3],
                                                       stencil.dHat);
                if (pfpx.valid)
                {
                    return rankOneHessianProduct<4, 9>(
                        stencil, pfpx.PFPx, pfpx.q0, stencilI5(stencil), direction);
                }
                break;
            }
        }

        return {
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0),
            glm::dvec3(0.0)
        };
    }
} // namespace ksk::engine::contact
