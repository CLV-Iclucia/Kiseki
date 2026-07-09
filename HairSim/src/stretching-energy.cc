#include "rod-energy-geometry.h"

#include <cmath>

namespace sim::hairsim {

StretchingEnergy::StretchingEnergy(const RodState& state,
                                   const RodRestState& rest,
                                   const RodMaterial& material)
    : state_(state), rest_(rest), material_(material) {}

void StretchingEnergy::accumulate(RodAssembly assembly) const {
  const size_t n = state_.size();
  const double axial_stiffness = material_.axialStiffness();
  for (size_t edge = 0; edge + 1 < n; ++edge) {
    const detail::EdgeGeometry geometry =
        detail::edgeGeometry(state_.blocks, edge);
    const double rest_length = rest_.metrics[edge].y;
    const double strain = geometry.length / rest_length - 1.0;
    assembly.energy.stretching +=
        0.5 * axial_stiffness / rest_length *
        std::pow(geometry.length - rest_length, 2);

    const detail::Vec3 edge_gradient =
        axial_stiffness * strain * geometry.tangent;
    const Eigen::Index first = static_cast<Eigen::Index>(4 * edge);
    const Eigen::Index second = first + 4;
    assembly.gradient.segment<3>(first) -= edge_gradient;
    assembly.gradient.segment<3>(second) += edge_gradient;

    const detail::Mat3 tangent_outer =
        geometry.tangent * geometry.tangent.transpose();
    const detail::Mat3 edge_hessian =
        axial_stiffness *
        ((1.0 / rest_length - 1.0 / geometry.length) *
             (detail::Mat3::Identity() - tangent_outer) +
         tangent_outer / rest_length);
    detail::addBlock(assembly.hessian, first, first, edge_hessian);
    detail::addBlock(assembly.hessian, first, second, -edge_hessian);
    detail::addBlock(assembly.hessian, second, first, -edge_hessian);
    detail::addBlock(assembly.hessian, second, second, edge_hessian);
  }
}

}  // namespace sim::hairsim
