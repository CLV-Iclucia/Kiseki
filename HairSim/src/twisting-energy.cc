#include "rod-energy-geometry.h"

namespace sim::hairsim {

TwistingEnergy::TwistingEnergy(
    const RodState& state, const RodRestState& rest,
    const RodMaterial& material)
    : state_(state), rest_(rest), material_(material) {}

void TwistingEnergy::accumulate(RodAssembly assembly) const {
  const size_t n = state_.size();
  for (size_t vertex = 1; vertex + 1 < n; ++vertex) {
    const double twist =
        state_.blocks[vertex].w - state_.blocks[vertex - 1].w -
        rest_.metrics[vertex].w;
    const double dual_length = rest_.metrics[vertex].z;
    const double twist_weight = material_.twistStiffness() / dual_length;

    detail::Vec12 twist_jacobian = detail::Vec12::Zero();
    twist_jacobian[3] = -1.0;
    twist_jacobian[7] = 1.0;

    // The Bishop frame is reconstructed for the current centerline, so the
    // reference-frame twist is a gauge choice rather than an explicit term.
    assembly.energy.twisting += 0.5 * twist_weight * twist * twist;

    const Eigen::Index offset =
        static_cast<Eigen::Index>(4 * (vertex - 1));
    detail::addLocalGradient(assembly.gradient, offset,
                             twist_weight * twist_jacobian * twist);
    detail::addLocalHessian(
        assembly.hessian, offset,
        twist_weight * twist_jacobian * twist_jacobian.transpose());
  }
}

}  // namespace sim::hairsim
