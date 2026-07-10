#include "rod-energy-geometry.h"

namespace ksk::hairsim {
namespace {

detail::Vec12 twistJacobian(const std::vector<RodBlock>& blocks, size_t vertex)
{
  const detail::EdgeGeometry previous =
      detail::edgeGeometry(blocks, vertex - 1);
  const detail::EdgeGeometry next = detail::edgeGeometry(blocks, vertex);
  const detail::Vec3 curvature_binormal =
      detail::curvatureBinormalGeometry(blocks, vertex, false).value;

  detail::Vec12 jacobian = detail::Vec12::Zero();
  jacobian.segment<3>(0) =
      -curvature_binormal / (2.0 * previous.length);
  jacobian[3] = -1.0;
  jacobian.segment<3>(4) =
      curvature_binormal / (2.0 * previous.length) -
      curvature_binormal / (2.0 * next.length);
  jacobian[7] = 1.0;
  jacobian.segment<3>(8) = curvature_binormal / (2.0 * next.length);
  return jacobian;
}

}  // namespace

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

    const detail::Vec12 twist_jacobian =
        twistJacobian(state_.blocks, vertex);

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

}  // namespace ksk::hairsim
