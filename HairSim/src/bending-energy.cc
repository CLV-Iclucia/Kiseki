#include "rod-energy-geometry.h"

namespace ksk::hairsim {

BendingEnergy::BendingEnergy(const RodState& state,
                             const RodRestState& rest,
                             const RodMaterial& material)
    : state_(state), rest_(rest), material_(material) {}

void BendingEnergy::accumulate(RodAssembly assembly) const {
  const size_t n = state_.size();
  for (size_t vertex = 1; vertex + 1 < n; ++vertex) {
    const size_t local = vertex - 1;
    const detail::CurvatureBinormalGeometry curvature =
        detail::curvatureBinormalGeometry(state_.blocks, vertex, true);
    const detail::Vec3 rest_curvature(rest_.curvature[local].x,
                                      rest_.curvature[local].y,
                                      rest_.curvature[local].z);
    const detail::Vec3 residual = curvature.value - rest_curvature;
    const double dual_length = rest_.metrics[vertex].z;
    const double bending_weight =
        material_.bendingStiffness() / dual_length;

    // Circular isotropic rod: bending depends on the curvature-binormal
    // residual, independent of a material-frame spin around the tangent.
    assembly.energy.bending +=
        0.5 * bending_weight * residual.squaredNorm();

    const detail::Vec12 bending_gradient =
        bending_weight * curvature.jacobian.transpose() * residual;
    const detail::Mat12 bending_hessian =
        bending_weight * curvature.jacobian.transpose() *
        curvature.jacobian;
    const Eigen::Index offset =
        static_cast<Eigen::Index>(4 * (vertex - 1));
    detail::addLocalGradient(assembly.gradient, offset, bending_gradient);
    detail::addLocalHessian(assembly.hessian, offset, bending_hessian);
  }
}

}  // namespace ksk::hairsim
