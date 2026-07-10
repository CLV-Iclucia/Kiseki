#pragma once

#include <HairSim/rod-energy.h>

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>

namespace ksk::hairsim::detail {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat3x12 = Eigen::Matrix<double, 3, 12>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;

struct EdgeGeometry {
  Vec3 edge;
  Vec3 tangent;
  double length;
};

struct CurvatureBinormalGeometry {
  Vec3 value = Vec3::Zero();
  Mat3x12 jacobian = Mat3x12::Zero();
};

inline Vec3 position(const RodBlock& block) {
  return {block.x, block.y, block.z};
}

inline Mat3 skew(const Vec3& v) {
  Mat3 result;
  result << 0.0, -v.z(), v.y(),
            v.z(), 0.0, -v.x(),
            -v.y(), v.x(), 0.0;
  return result;
}

inline EdgeGeometry edgeGeometry(const std::vector<RodBlock>& blocks,
                                 size_t edge) {
  EdgeGeometry result;
  result.edge = position(blocks.at(edge + 1)) - position(blocks.at(edge));
  result.length = result.edge.norm();
  if (result.length < 1e-10)
    throw std::runtime_error("rod contains a near-zero length edge");
  result.tangent = result.edge / result.length;
  return result;
}

inline CurvatureBinormalGeometry curvatureBinormalGeometry(
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

inline void addBlock(Eigen::MatrixXd& matrix, Eigen::Index row,
                     Eigen::Index column, const Mat3& block) {
  matrix.block<3, 3>(row, column) += block;
}

inline void addLocalGradient(Eigen::VectorXd& gradient, Eigen::Index offset,
                             const Vec12& local) {
  gradient.segment<12>(offset) += local;
}

inline void addLocalHessian(Eigen::MatrixXd& hessian, Eigen::Index offset,
                            const Mat12& local) {
  hessian.block<12, 12>(offset, offset) += local;
}

}  // namespace ksk::hairsim::detail
