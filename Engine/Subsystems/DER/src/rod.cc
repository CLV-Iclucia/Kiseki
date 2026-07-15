#include <DER/rod.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace ksk::der {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using Mat3x12 = Eigen::Matrix<double, 3, 12>;

struct EdgeData {
  Vec3 vector;
  Vec3 tangent;
  double length = 0.0;
};

struct CurvatureData {
  Vec3 value = Vec3::Zero();
  Mat3x12 derivative = Mat3x12::Zero();
};

Vec3 xyz(const RodDof& block)
{
  return {block.x, block.y, block.z};
}

glm::dvec3 toGlm(const Vec3& value)
{
  return {value.x(), value.y(), value.z()};
}

Vec3 toEigen(const glm::dvec3& value)
{
  return {value.x, value.y, value.z};
}

Mat3 crossMatrix(const Vec3& value)
{
  Mat3 matrix;
  matrix << 0.0, -value.z(), value.y(),
            value.z(), 0.0, -value.x(),
            -value.y(), value.x(), 0.0;
  return matrix;
}

EdgeData edgeData(std::span<const RodDof> blocks, size_t edge)
{
  EdgeData data;
  data.vector = xyz(blocks[edge + 1]) - xyz(blocks[edge]);
  data.length = data.vector.norm();
  if (data.length < 1e-10) {
    throw std::runtime_error("DER rod contains a near-zero length edge");
  }
  data.tangent = data.vector / data.length;
  return data;
}

Vec3 fallbackDirector(const Vec3& tangent)
{
  const Vec3 axis = std::abs(tangent.z()) < 0.9 ? Vec3::UnitZ()
                                                : Vec3::UnitY();
  const Vec3 director = axis.cross(tangent);
  const double norm = director.norm();
  if (norm < 1e-8) {
    throw std::runtime_error("DER rod reference director is undefined");
  }
  return director / norm;
}

Vec3 normalizeDirector(const Vec3& director, const Vec3& tangent)
{
  const Vec3 projected = director - tangent * director.dot(tangent);
  const double norm = projected.norm();
  if (norm < 1e-8) {
    return fallbackDirector(tangent);
  }
  return projected / norm;
}

Vec3 transportDirector(const Vec3& director, const Vec3& from, const Vec3& to)
{
  const Vec3 axis = from.cross(to);
  const double denominator = 1.0 + from.dot(to);
  if (denominator < 1e-7) {
    return fallbackDirector(to);
  }
  const Vec3 transported =
      director + axis.cross(director) +
      axis.cross(axis.cross(director)) / denominator;
  return transported.normalized();
}

double signedAngle(const Vec3& from, const Vec3& to, const Vec3& axis)
{
  return std::atan2(axis.dot(from.cross(to)), from.dot(to));
}

CurvatureData curvatureBinormal(std::span<const RodDof> blocks,
                                size_t vertex,
                                bool withDerivative)
{
  const EdgeData left = edgeData(blocks, vertex - 1);
  const EdgeData right = edgeData(blocks, vertex);
  const Vec3& t0 = left.tangent;
  const Vec3& t1 = right.tangent;
  const double chi = 1.0 + t0.dot(t1);
  if (chi < 1e-7) {
    throw std::runtime_error("DER curvature is singular for antiparallel edges");
  }

  CurvatureData data;
  const Vec3 cross = t0.cross(t1);
  data.value = 2.0 * cross / chi;
  if (!withDerivative) {
    return data;
  }

  const Mat3 project0 = (Mat3::Identity() - t0 * t0.transpose()) / left.length;
  const Mat3 project1 = (Mat3::Identity() - t1 * t1.transpose()) / right.length;
  const Mat3 dkb_dt0 =
      2.0 * (-crossMatrix(t1) / chi -
             cross * t1.transpose() / (chi * chi));
  const Mat3 dkb_dt1 =
      2.0 * (crossMatrix(t0) / chi -
             cross * t0.transpose() / (chi * chi));
  const Mat3 dkb_de0 = dkb_dt0 * project0;
  const Mat3 dkb_de1 = dkb_dt1 * project1;

  data.derivative.block<3, 3>(0, 0) = -dkb_de0;
  data.derivative.block<3, 3>(0, 4) = dkb_de0 - dkb_de1;
  data.derivative.block<3, 3>(0, 8) = dkb_de1;
  return data;
}

std::vector<RodDof> toBlocks(const Eigen::VectorXd& values)
{
  std::vector<RodDof> blocks(static_cast<size_t>(values.size() / 4));
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (int lane = 0; lane < 4; ++lane) {
      blocks[i][lane] = values[static_cast<Eigen::Index>(4 * i + lane)];
    }
  }
  return blocks;
}

std::vector<glm::dvec3> transportedDirectors(
    const std::vector<glm::dvec3>& directors,
    const RodState& previous,
    const RodState& current)
{
  std::vector<glm::dvec3> result(directors.size());
  for (size_t edge = 0; edge + 1 < current.size(); ++edge) {
    const EdgeData old_edge = edgeData(previous.blocks, edge);
    const EdgeData new_edge = edgeData(current.blocks, edge);
    const Vec3 old_director =
        normalizeDirector(toEigen(directors.at(edge)), old_edge.tangent);
    const Vec3 moved =
        transportDirector(old_director, old_edge.tangent, new_edge.tangent);
    result[edge] = toGlm(normalizeDirector(moved, new_edge.tangent));
  }
  return result;
}

double referenceTwistInGauge(const RodState& state,
                             const std::vector<glm::dvec3>& directors,
                             size_t vertex)
{
  if (vertex == 0 || vertex + 1 >= state.size()) {
    return 0.0;
  }
  const EdgeData left = edgeData(state.blocks, vertex - 1);
  const EdgeData right = edgeData(state.blocks, vertex);
  const Vec3 left_director =
      normalizeDirector(toEigen(directors.at(vertex - 1)), left.tangent);
  const Vec3 right_director =
      normalizeDirector(toEigen(directors.at(vertex)), right.tangent);
  const Vec3 transported =
      transportDirector(left_director, left.tangent, right.tangent);
  return signedAngle(transported, right_director, right.tangent);
}

void addBlock(Eigen::MatrixXd& matrix,
              Eigen::Index row,
              Eigen::Index column,
              const Mat3& block)
{
  matrix.block<3, 3>(row, column) += block;
}

void addLocal(Eigen::VectorXd& vector,
              Eigen::Index offset,
              const Vec12& local)
{
  vector.segment<12>(offset) += local;
}

void addLocal(Eigen::MatrixXd& matrix,
              Eigen::Index offset,
              const Mat12& local)
{
  matrix.block<12, 12>(offset, offset) += local;
}

Vec12 twistDerivative(std::span<const RodDof> blocks, size_t vertex)
{
  const EdgeData left = edgeData(blocks, vertex - 1);
  const EdgeData right = edgeData(blocks, vertex);
  const Vec3 kb = curvatureBinormal(blocks, vertex, false).value;

  Vec12 derivative = Vec12::Zero();
  derivative.segment<3>(0) = -kb / (2.0 * left.length);
  derivative[3] = -1.0;
  derivative.segment<3>(4) =
      kb / (2.0 * left.length) - kb / (2.0 * right.length);
  derivative[7] = 1.0;
  derivative.segment<3>(8) = kb / (2.0 * right.length);
  return derivative;
}

void accumulateStretching(const RodState& state,
                          const RodRestState& rest,
                          const RodMaterial& material,
                          RodEnergyComponents& energy,
                          Eigen::VectorXd& gradient,
                          Eigen::MatrixXd& hessian)
{
  const double stiffness = material.axialStiffness();
  for (size_t edge = 0; edge + 1 < state.size(); ++edge) {
    const EdgeData geometry = edgeData(state.blocks, edge);
    const double rest_length = rest.metrics[edge].y;
    const double strain = geometry.length / rest_length - 1.0;
    const double extension = geometry.length - rest_length;

    energy.stretching +=
        0.5 * stiffness * extension * extension / rest_length;

    const Vec3 edge_gradient = stiffness * strain * geometry.tangent;
    const Eigen::Index a = static_cast<Eigen::Index>(4 * edge);
    const Eigen::Index b = a + 4;
    gradient.segment<3>(a) -= edge_gradient;
    gradient.segment<3>(b) += edge_gradient;

    const Mat3 tangent_outer = geometry.tangent * geometry.tangent.transpose();
    const Mat3 edge_hessian =
        stiffness *
        ((1.0 / rest_length - 1.0 / geometry.length) *
             (Mat3::Identity() - tangent_outer) +
         tangent_outer / rest_length);
    addBlock(hessian, a, a, edge_hessian);
    addBlock(hessian, a, b, -edge_hessian);
    addBlock(hessian, b, a, -edge_hessian);
    addBlock(hessian, b, b, edge_hessian);
  }
}

void accumulateBending(const RodState& state,
                       const RodRestState& rest,
                       const RodMaterial& material,
                       RodEnergyComponents& energy,
                       Eigen::VectorXd& gradient,
                       Eigen::MatrixXd& hessian)
{
  for (size_t vertex = 1; vertex + 1 < state.size(); ++vertex) {
    const CurvatureData curvature =
        curvatureBinormal(state.blocks, vertex, true);
    const glm::dvec4& rest_value = rest.curvature[vertex - 1];
    const Vec3 rest_curvature(rest_value.x, rest_value.y, rest_value.z);
    const Vec3 residual = curvature.value - rest_curvature;
    const double weight = material.bendingStiffness() / rest.metrics[vertex].z;

    energy.bending += 0.5 * weight * residual.squaredNorm();

    const Vec12 local_gradient =
        weight * curvature.derivative.transpose() * residual;
    const Mat12 local_hessian =
        weight * curvature.derivative.transpose() * curvature.derivative;
    const Eigen::Index offset = static_cast<Eigen::Index>(4 * (vertex - 1));
    addLocal(gradient, offset, local_gradient);
    addLocal(hessian, offset, local_hessian);
  }
}

void accumulateTwisting(const RodState& state,
                        const RodRestState& rest,
                        const RodMaterial& material,
                        const std::vector<double>& reference_twists,
                        RodEnergyComponents& energy,
                        Eigen::VectorXd& gradient,
                        Eigen::MatrixXd& hessian)
{
  for (size_t vertex = 1; vertex + 1 < state.size(); ++vertex) {
    const double twist =
        state.blocks[vertex].w - state.blocks[vertex - 1].w +
        reference_twists[vertex] - rest.metrics[vertex].w;
    const double weight = material.twistStiffness() / rest.metrics[vertex].z;
    const Vec12 derivative = twistDerivative(state.blocks, vertex);

    energy.twisting += 0.5 * weight * twist * twist;

    const Eigen::Index offset = static_cast<Eigen::Index>(4 * (vertex - 1));
    addLocal(gradient, offset, weight * derivative * twist);
    addLocal(hessian, offset, weight * derivative * derivative.transpose());
  }
}

void accumulateGravity(const RodState& state,
                       const Eigen::VectorXd& mass,
                       const glm::dvec3& gravity,
                       RodEnergyComponents& energy,
                       Eigen::VectorXd& gradient)
{
  const Vec3 g(gravity.x, gravity.y, gravity.z);
  for (size_t vertex = 0; vertex < state.size(); ++vertex) {
    const Eigen::Index offset = static_cast<Eigen::Index>(4 * vertex);
    const double m = mass[offset];
    energy.gravity -= m * g.dot(xyz(state.blocks[vertex]));
    gradient.segment<3>(offset) -= m * g;
  }
}

void accumulateConstraint(const RodState& state,
                          const RodConstraint& constraint,
                          RodEnergyComponents& energy,
                          Eigen::VectorXd& gradient,
                          Eigen::MatrixXd& hessian)
{
  if (constraint.sample >= state.size()) {
    throw std::runtime_error("DER rod constraint sample is out of range");
  }
  const Eigen::Index offset =
      static_cast<Eigen::Index>(4 * constraint.sample);

  int lane = -1;
  switch (constraint.property) {
    case RodConstraintProperty::X:
      lane = 0;
      break;
    case RodConstraintProperty::Y:
      lane = 1;
      break;
    case RodConstraintProperty::Z:
      lane = 2;
      break;
    case RodConstraintProperty::Twist:
      if (constraint.sample + 1 >= state.size()) {
        throw std::runtime_error(
            "DER rod twist constraint sample is out of range");
      }
      lane = 3;
      break;
  }

  const double delta =
      state.blocks[constraint.sample][lane] - constraint.target;
  const double constraint_energy =
      0.5 * constraint.stiffness * delta * delta;
  const double gradient_contribution = constraint.stiffness * delta;
  const double hessian_contribution = constraint.stiffness;

  energy.constraint += constraint_energy;
  gradient[offset + lane] += gradient_contribution;
  hessian(offset + lane, offset + lane) += hessian_contribution;
}

}  // namespace

glm::dvec3 RodState::position(size_t index) const
{
  const RodDof& block = blocks[index];
  return {block.x, block.y, block.z};
}

void RodState::setPosition(size_t index, const glm::dvec3& value)
{
  RodDof& block = blocks[index];
  block.x = value.x;
  block.y = value.y;
  block.z = value.z;
}

double RodState::theta(size_t index) const
{
  return blocks[index].w;
}

void RodState::setTheta(size_t index, double value)
{
  blocks[index].w = value;
}

double RodMaterial::area() const noexcept
{
  return std::numbers::pi * radius * radius;
}

double RodMaterial::areaMoment() const noexcept
{
  return 0.25 * std::numbers::pi * std::pow(radius, 4);
}

double RodMaterial::polarMoment() const noexcept
{
  return 2.0 * areaMoment();
}

double RodMaterial::axialStiffness() const noexcept
{
  return youngsModulus * area();
}

double RodMaterial::bendingStiffness() const noexcept
{
  return youngsModulus * areaMoment();
}

double RodMaterial::twistStiffness() const noexcept
{
  return shearModulus * polarMoment();
}

double RodEnergyComponents::total() const noexcept
{
  return stretching + bending + twisting + gravity + constraint;
}

Rod::Rod(std::vector<RodDof> restBlocks, RodMaterial material)
    : material_(material)
{
  if (restBlocks.size() < 3) {
    throw std::invalid_argument("a DER rod requires at least three vertices");
  }

  restBlocks.back().w = 0.0;
  state_storage_ = restBlocks;
  velocity_storage_.assign(restBlocks.size(), RodDof(0.0));
  bindOwnedState();
  rest_.blocks = restBlocks;
  reference_directors_.resize(restBlocks.size() - 1);
  rest_.metrics.assign(restBlocks.size(), glm::dvec4(0.0));
  rest_.curvature.reserve(restBlocks.size() - 2);

  for (size_t edge = 0; edge + 1 < restBlocks.size(); ++edge) {
    const double length = edgeData(restBlocks, edge).length;
    rest_.metrics[edge].y = length;
    rest_.metrics[edge + 1].x = length;
  }
  for (size_t vertex = 0; vertex < restBlocks.size(); ++vertex) {
    rest_.metrics[vertex].z =
        0.5 * (rest_.metrics[vertex].x + rest_.metrics[vertex].y);
  }
  for (size_t vertex = 1; vertex + 1 < restBlocks.size(); ++vertex) {
    const Vec3 curvature = curvatureBinormal(restBlocks, vertex, false).value;
    rest_.curvature.emplace_back(curvature.x(), curvature.y(), curvature.z(),
                                 0.0);
  }

  resetReferenceFrames();
  for (size_t vertex = 1; vertex + 1 < restBlocks.size(); ++vertex) {
    rest_.metrics[vertex].w =
        restBlocks[vertex].w - restBlocks[vertex - 1].w +
        referenceTwist(vertex);
  }
}

Rod::Rod(const Rod& other)
    : state_storage_(other.state_.blocks.begin(), other.state_.blocks.end()),
      velocity_storage_(other.velocity_.blocks.begin(),
                        other.velocity_.blocks.end()),
      rest_(other.rest_),
      reference_directors_(other.reference_directors_),
      material_(other.material_),
      pin_root_position_(other.pin_root_position_),
      pin_tip_position_(other.pin_tip_position_),
      terminal_theta_target_(other.terminal_theta_target_),
      constraints_(other.constraints_)
{
  bindOwnedState();
}

Rod::Rod(Rod&& other) noexcept
    : state_storage_(other.state_.blocks.begin(), other.state_.blocks.end()),
      velocity_storage_(other.velocity_.blocks.begin(),
                        other.velocity_.blocks.end()),
      rest_(std::move(other.rest_)),
      reference_directors_(std::move(other.reference_directors_)),
      material_(other.material_),
      pin_root_position_(other.pin_root_position_),
      pin_tip_position_(other.pin_tip_position_),
      terminal_theta_target_(std::move(other.terminal_theta_target_)),
      constraints_(std::move(other.constraints_))
{
  bindOwnedState();
}

Rod& Rod::operator=(const Rod& other)
{
  if (this == &other) {
    return *this;
  }
  state_storage_.assign(other.state_.blocks.begin(), other.state_.blocks.end());
  velocity_storage_.assign(other.velocity_.blocks.begin(),
                           other.velocity_.blocks.end());
  rest_ = other.rest_;
  reference_directors_ = other.reference_directors_;
  material_ = other.material_;
  pin_root_position_ = other.pin_root_position_;
  pin_tip_position_ = other.pin_tip_position_;
  terminal_theta_target_ = other.terminal_theta_target_;
  constraints_ = other.constraints_;
  bindOwnedState();
  return *this;
}

Rod& Rod::operator=(Rod&& other) noexcept
{
  if (this == &other) {
    return *this;
  }
  state_storage_.assign(other.state_.blocks.begin(), other.state_.blocks.end());
  velocity_storage_.assign(other.velocity_.blocks.begin(),
                           other.velocity_.blocks.end());
  rest_ = std::move(other.rest_);
  reference_directors_ = std::move(other.reference_directors_);
  material_ = other.material_;
  pin_root_position_ = other.pin_root_position_;
  pin_tip_position_ = other.pin_tip_position_;
  terminal_theta_target_ = std::move(other.terminal_theta_target_);
  constraints_ = std::move(other.constraints_);
  bindOwnedState();
  return *this;
}

void Rod::resetReferenceFrames()
{
  if (state_.size() < 2) {
    return;
  }

  reference_directors_.resize(state_.size() - 1);
  EdgeData previous = edgeData(state_.blocks, 0);
  reference_directors_[0] = toGlm(fallbackDirector(previous.tangent));
  for (size_t edge = 1; edge + 1 < state_.size(); ++edge) {
    const EdgeData current = edgeData(state_.blocks, edge);
    const Vec3 old_director =
        normalizeDirector(toEigen(reference_directors_[edge - 1]),
                          previous.tangent);
    const Vec3 moved =
        transportDirector(old_director, previous.tangent, current.tangent);
    reference_directors_[edge] =
        toGlm(normalizeDirector(moved, current.tangent));
    previous = current;
  }
}

void Rod::transportReferenceFrames(const RodState& previousState)
{
  if (state_.size() < 2) {
    return;
  }
  if (reference_directors_.size() != state_.size() - 1) {
    resetReferenceFrames();
  }
  reference_directors_ =
      transportedDirectors(reference_directors_, previousState, state_);
}

double Rod::referenceTwist(size_t vertex) const
{
  if (vertex == 0 || vertex + 1 >= state_.size()) {
    return 0.0;
  }
  if (reference_directors_.size() != state_.size() - 1) {
    throw std::runtime_error("DER rod reference frames are not initialized");
  }
  return referenceTwistInGauge(state_, reference_directors_, vertex);
}

double Rod::materialTwist(size_t vertex) const
{
  if (vertex == 0 || vertex + 1 >= state_.size()) {
    return 0.0;
  }
  return state_.blocks[vertex].w - state_.blocks[vertex - 1].w +
         referenceTwist(vertex);
}

void Rod::setRootPositionPinned(bool pinned) noexcept
{
  pin_root_position_ = pinned;
}

void Rod::setTipPositionPinned(bool pinned) noexcept
{
  pin_tip_position_ = pinned;
}

void Rod::setTerminalThetaTarget(std::optional<double> target) noexcept
{
  terminal_theta_target_ = target;
}

void Rod::addConstraint(RodConstraint constraint)
{
  constraints_.push_back(constraint);
}

void Rod::clearConstraints() noexcept
{
  constraints_.clear();
}

std::span<const RodConstraint> Rod::constraints() const noexcept
{
  return constraints_;
}

void Rod::bindState(std::span<RodDof> stateBlocks,
                    std::span<RodDof> velocityBlocks)
{
  if (stateBlocks.size() != rest_.blocks.size() ||
      velocityBlocks.size() != rest_.blocks.size()) {
    throw std::invalid_argument("DER rod state binding has an invalid size");
  }
  state_.blocks = stateBlocks;
  velocity_.blocks = velocityBlocks;
}

void Rod::bindOwnedState() noexcept
{
  state_.blocks =
      std::span<RodDof>(state_storage_.data(), state_storage_.size());
  velocity_.blocks =
      std::span<RodDof>(velocity_storage_.data(), velocity_storage_.size());
}

RodEvaluation Rod::evaluate(const glm::dvec3& gravity) const
{
  std::vector<double> reference_twists(state_.size(), 0.0);
  for (size_t vertex = 1; vertex + 1 < state_.size(); ++vertex) {
    reference_twists[vertex] = referenceTwist(vertex);
  }
  return evaluate(gravity, reference_twists);
}

RodEvaluation Rod::evaluate(const glm::dvec3& gravity,
                            const RodState& previousState) const
{
  const std::vector<glm::dvec3> directors =
      transportedDirectors(reference_directors_, previousState, state_);
  std::vector<double> reference_twists(state_.size(), 0.0);
  for (size_t vertex = 1; vertex + 1 < state_.size(); ++vertex) {
    reference_twists[vertex] =
        referenceTwistInGauge(state_, directors, vertex);
  }
  return evaluate(gravity, reference_twists);
}

RodEvaluation Rod::evaluate(const glm::dvec3& gravity,
                            const std::vector<double>& referenceTwists) const
{
  RodEvaluation result;
  const Eigen::Index dofs = static_cast<Eigen::Index>(4 * state_.size());
  result.gradient.assign(state_.size(), RodDof(0.0));
  result.hessian.resize(dofs, dofs);

  try {
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dofs);
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dofs, dofs);

    accumulateStretching(state_, rest_, material_, result.energy, gradient,
                         hessian);
    accumulateBending(state_, rest_, material_, result.energy, gradient,
                      hessian);
    accumulateTwisting(state_, rest_, material_, referenceTwists, result.energy,
                       gradient, hessian);
    accumulateGravity(state_, massDiagonal(), gravity, result.energy, gradient);
    for (const RodConstraint& constraint : constraints_) {
      accumulateConstraint(state_, constraint, result.energy, gradient,
                           hessian);
    }

    result.gradient = toBlocks(gradient);
    result.hessian = hessian.sparseView(0.0, 1e-14);
    result.valid = true;
  } catch (const std::exception& exception) {
    result.diagnostic = exception.what();
  }
  return result;
}

Eigen::VectorXd Rod::massDiagonal() const
{
  const Eigen::Index dofs = static_cast<Eigen::Index>(4 * state_.size());
  Eigen::VectorXd mass = Eigen::VectorXd::Zero(dofs);
  for (size_t vertex = 0; vertex < state_.size(); ++vertex) {
    const double dual_length = rest_.metrics[vertex].z;
    const double translational =
        material_.density * material_.area() * dual_length;
    mass.segment<3>(static_cast<Eigen::Index>(4 * vertex))
        .setConstant(translational);
    if (vertex + 1 < state_.size()) {
      mass[static_cast<Eigen::Index>(4 * vertex + 3)] =
          material_.density * material_.polarMoment() *
          rest_.metrics[vertex].y;
    }
  }
  mass[static_cast<Eigen::Index>(4 * state_.size() - 1)] = 1.0;
  return mass;
}

}  // namespace ksk::der
