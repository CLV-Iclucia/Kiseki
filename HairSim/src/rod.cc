#include <HairSim/rod.h>
#include <HairSim/rod-energy.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace ksk::hairsim {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat3x12 = Eigen::Matrix<double, 3, 12>;

struct EdgeGeometry {
  Vec3 edge;
  Vec3 tangent;
  double length;
};

struct CurvatureBinormalGeometry {
  Vec3 value = Vec3::Zero();
  Mat3x12 jacobian = Mat3x12::Zero();
};

Vec3 position(const RodBlock& block) {
  return {block.x, block.y, block.z};
}

Mat3 skew(const Vec3& v) {
  Mat3 result;
  result << 0.0, -v.z(), v.y(),
            v.z(), 0.0, -v.x(),
            -v.y(), v.x(), 0.0;
  return result;
}

glm::dvec3 toGlm(const Vec3& value) {
  return {value.x(), value.y(), value.z()};
}

Vec3 toEigen(const glm::dvec3& value) {
  return {value.x, value.y, value.z};
}

EdgeGeometry edgeGeometry(const std::vector<RodBlock>& blocks, size_t edge) {
  EdgeGeometry result;
  result.edge = position(blocks.at(edge + 1)) - position(blocks.at(edge));
  result.length = result.edge.norm();
  if (result.length < 1e-10)
    throw std::runtime_error("rod contains a near-zero length edge");
  result.tangent = result.edge / result.length;
  return result;
}

Vec3 fallbackDirector(const Vec3& tangent) {
  const Vec3 axis = std::abs(tangent.z()) < 0.9 ? Vec3::UnitZ()
                                                : Vec3::UnitY();
  const Vec3 director = axis.cross(tangent);
  const double length = director.norm();
  if (length < 1e-8)
    throw std::runtime_error("rod reference director is undefined");
  return director / length;
}

Vec3 parallelTransport(const Vec3& director, const Vec3& from,
                       const Vec3& to) {
  const Vec3 axis = from.cross(to);
  const double denominator = 1.0 + from.dot(to);
  if (denominator < 1e-7) return fallbackDirector(to);
  return (director + axis.cross(director) +
          axis.cross(axis.cross(director)) / denominator).normalized();
}

Vec3 orthonormalizeDirector(const Vec3& director, const Vec3& tangent) {
  Vec3 result = director - tangent * director.dot(tangent);
  const double length = result.norm();
  if (length < 1e-8) return fallbackDirector(tangent);
  return result / length;
}

CurvatureBinormalGeometry curvatureBinormalGeometry(
    const std::vector<RodBlock>& blocks, size_t vertex,
    bool compute_jacobian) {
  const EdgeGeometry previous = edgeGeometry(blocks, vertex - 1);
  const EdgeGeometry next = edgeGeometry(blocks, vertex);
  const Vec3& t0 = previous.tangent;
  const Vec3& t1 = next.tangent;
  const double chi = 1.0 + t0.dot(t1);
  if (chi < 1e-7)
    throw std::runtime_error(
        "curvature is singular at an antiparallel edge pair");

  CurvatureBinormalGeometry result;
  const Vec3 cross_t = t0.cross(t1);
  result.value = 2.0 * cross_t / chi;
  if (!compute_jacobian) return result;

  const Mat3 projection0 =
      (Mat3::Identity() - t0 * t0.transpose()) / previous.length;
  const Mat3 projection1 =
      (Mat3::Identity() - t1 * t1.transpose()) / next.length;
  const Mat3 dkb_dt0 =
      2.0 * (-skew(t1) / chi -
             cross_t * t1.transpose() / (chi * chi));
  const Mat3 dkb_dt1 =
      2.0 * (skew(t0) / chi -
             cross_t * t0.transpose() / (chi * chi));
  const Mat3 dkb_de0 = dkb_dt0 * projection0;
  const Mat3 dkb_de1 = dkb_dt1 * projection1;

  result.jacobian.block<3, 3>(0, 0) = -dkb_de0;
  result.jacobian.block<3, 3>(0, 4) = dkb_de0 - dkb_de1;
  result.jacobian.block<3, 3>(0, 8) = dkb_de1;
  return result;
}

std::vector<RodBlock> unflatten(const Eigen::VectorXd& values) {
  std::vector<RodBlock> result(static_cast<size_t>(values.size() / 4));
  for (size_t i = 0; i < result.size(); ++i)
    for (int lane = 0; lane < 4; ++lane)
      result[i][lane] = values[static_cast<Eigen::Index>(4 * i + lane)];
  return result;
}

void accumulateGravityEnergy(const RodState& state,
                             const Eigen::VectorXd& mass,
                             const glm::dvec3& gravity,
                             RodAssembly assembly) {
  const Vec3 gravity_vector(gravity.x, gravity.y, gravity.z);
  for (size_t vertex = 0; vertex < state.size(); ++vertex) {
    const Eigen::Index offset = static_cast<Eigen::Index>(4 * vertex);
    const double vertex_mass = mass[offset];
    const Vec3 current = position(state.blocks[vertex]);
    assembly.energy.gravity -= vertex_mass * gravity_vector.dot(current);
    assembly.gradient.segment<3>(offset) -= vertex_mass * gravity_vector;
  }
}

void accumulateRootConstraintEnergy(const RodState& state,
                                    const RodRestState& rest,
                                    const RodMaterial& material,
                                    RodAssembly assembly) {
  const Vec3 root_offset =
      position(state.blocks.front()) - position(rest.blocks.front());
  assembly.energy.constraint +=
      0.5 * material.rootStiffness * root_offset.squaredNorm();
  assembly.gradient.segment<3>(0) += material.rootStiffness * root_offset;
  assembly.hessian.block<3, 3>(0, 0).diagonal().array() +=
      material.rootStiffness;

  const Vec3 root_edge_offset =
      (position(state.blocks[1]) - position(state.blocks[0])) -
      (position(rest.blocks[1]) - position(rest.blocks[0]));
  assembly.energy.constraint +=
      0.5 * material.rootStiffness * root_edge_offset.squaredNorm();
  assembly.gradient.segment<3>(0) -=
      material.rootStiffness * root_edge_offset;
  assembly.gradient.segment<3>(4) +=
      material.rootStiffness * root_edge_offset;
  assembly.hessian.block<3, 3>(0, 0).diagonal().array() +=
      material.rootStiffness;
  assembly.hessian.block<3, 3>(0, 4).diagonal().array() -=
      material.rootStiffness;
  assembly.hessian.block<3, 3>(4, 0).diagonal().array() -=
      material.rootStiffness;
  assembly.hessian.block<3, 3>(4, 4).diagonal().array() +=
      material.rootStiffness;

  if (material.pinRootTwist) {
    const double twist = state.blocks.front().w - rest.blocks.front().w;
    assembly.energy.constraint +=
        0.5 * material.rootStiffness * twist * twist;
    assembly.gradient[3] += material.rootStiffness * twist;
    assembly.hessian(3, 3) += material.rootStiffness;
  }
}

}  // namespace

glm::dvec3 RodState::position(size_t i) const {
  const auto& block = blocks.at(i);
  return {block.x, block.y, block.z};
}

void RodState::setPosition(size_t i, const glm::dvec3& value) {
  auto& block = blocks.at(i);
  block.x = value.x;
  block.y = value.y;
  block.z = value.z;
}

double RodState::theta(size_t i) const { return blocks.at(i).w; }

void RodState::setTheta(size_t i, double value) { blocks.at(i).w = value; }

double RodMaterial::area() const noexcept {
  return std::numbers::pi * radius * radius;
}

double RodMaterial::areaMoment() const noexcept {
  return 0.25 * std::numbers::pi * std::pow(radius, 4);
}

double RodMaterial::polarMoment() const noexcept {
  return 2.0 * areaMoment();
}

double RodMaterial::axialStiffness() const noexcept {
  return youngsModulus * area();
}

double RodMaterial::bendingStiffness() const noexcept {
  return youngsModulus * areaMoment();
}

double RodMaterial::twistStiffness() const noexcept {
  return shearModulus * polarMoment();
}

double RodEnergyComponents::total() const noexcept {
  return stretching + bending + twisting + gravity + constraint;
}

Rod::Rod(std::vector<RodBlock> restBlocks, RodMaterial material)
    : material_(material) {
  if (restBlocks.size() < 3)
    throw std::invalid_argument("a rod requires at least three vertices");
  restBlocks.back().w = 0.0;
  state_.blocks = restBlocks;
  velocity_.blocks.assign(restBlocks.size(), RodBlock(0.0));
  rest_.blocks = restBlocks;
  reference_directors_.resize(restBlocks.size() - 1);
  rest_.curvature.reserve(restBlocks.size() - 2);
  rest_.metrics.resize(restBlocks.size(), glm::dvec4(0.0));

  for (size_t edge = 0; edge + 1 < restBlocks.size(); ++edge) {
    const double length = edgeGeometry(restBlocks, edge).length;
    rest_.metrics[edge].y = length;
    rest_.metrics[edge + 1].x = length;
  }
  for (size_t i = 0; i < restBlocks.size(); ++i)
    rest_.metrics[i].z =
        0.5 * (rest_.metrics[i].x + rest_.metrics[i].y);

  for (size_t vertex = 1; vertex + 1 < restBlocks.size(); ++vertex) {
    const Vec3 curvature =
        curvatureBinormalGeometry(restBlocks, vertex, false).value;
    rest_.curvature.emplace_back(
        curvature.x(), curvature.y(), curvature.z(), 0.0);
    rest_.metrics[vertex].w =
        restBlocks[vertex].w - restBlocks[vertex - 1].w;
  }
  resetReferenceFrames();
}

void Rod::resetReferenceFrames() {
  if (state_.blocks.size() < 2) return;
  reference_directors_.resize(state_.blocks.size() - 1);
  EdgeGeometry previous = edgeGeometry(state_.blocks, 0);
  reference_directors_[0] = toGlm(fallbackDirector(previous.tangent));
  for (size_t edge = 1; edge + 1 < state_.blocks.size(); ++edge) {
    const EdgeGeometry current = edgeGeometry(state_.blocks, edge);
    const Vec3 old_director =
        orthonormalizeDirector(toEigen(reference_directors_[edge - 1]),
                               previous.tangent);
    const Vec3 transported =
        parallelTransport(old_director, previous.tangent, current.tangent);
    reference_directors_[edge] =
        toGlm(orthonormalizeDirector(transported, current.tangent));
    previous = current;
  }
}

void Rod::transportReferenceFrames(const RodState& previous_state) {
  if (state_.blocks.size() < 2) return;
  if (reference_directors_.size() != state_.blocks.size() - 1)
    resetReferenceFrames();
  for (size_t edge = 0; edge + 1 < state_.blocks.size(); ++edge) {
    const EdgeGeometry previous = edgeGeometry(previous_state.blocks, edge);
    const EdgeGeometry current = edgeGeometry(state_.blocks, edge);
    const Vec3 old_director =
        orthonormalizeDirector(toEigen(reference_directors_[edge]),
                               previous.tangent);
    const Vec3 transported =
        parallelTransport(old_director, previous.tangent, current.tangent);
    reference_directors_[edge] =
        toGlm(orthonormalizeDirector(transported, current.tangent));
  }
}

RodEvaluation Rod::evaluate(const glm::dvec3& gravity) const {
  RodEvaluation result;
  const size_t n = state_.size();
  const Eigen::Index dofs = static_cast<Eigen::Index>(4 * n);
  result.gradient.assign(n, RodBlock(0.0));
  result.hessian.resize(dofs, dofs);

  try {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dofs);
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dofs, dofs);
    RodAssembly assembly{result.energy, gradient, hessian};

    StretchingEnergy(state_, rest_, material_).accumulate(assembly);
    BendingEnergy(state_, rest_, material_).accumulate(assembly);
    TwistingEnergy(state_, rest_, material_).accumulate(assembly);
    accumulateGravityEnergy(state_, massDiagonal(), gravity, assembly);
    accumulateRootConstraintEnergy(state_, rest_, material_, assembly);

    result.gradient = unflatten(gradient);
    result.hessian = hessian.sparseView(0.0, 1e-14);
    result.valid = true;
  } catch (const std::exception& exception) {
    result.diagnostic = exception.what();
  }
  return result;
}

Eigen::VectorXd Rod::massDiagonal() const {
  const size_t n = state_.size();
  Eigen::VectorXd mass =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(4 * n));
  for (size_t i = 0; i < n; ++i) {
    const double dual_length = rest_.metrics[i].z;
    const double translational =
        material_.density * material_.area() * dual_length;
    mass.segment<3>(static_cast<Eigen::Index>(4 * i))
        .setConstant(translational);
    if (i + 1 < n) {
      mass[static_cast<Eigen::Index>(4 * i + 3)] =
          material_.density * material_.polarMoment() *
          rest_.metrics[i].y;
    }
  }
  mass[static_cast<Eigen::Index>(4 * n - 1)] = 1.0;
  return mass;
}

}  // namespace ksk::hairsim
