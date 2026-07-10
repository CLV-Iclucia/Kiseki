//
// Created by creeper on 23-8-9.
//

#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#include <HairSim/hair-sim.h>
#include <vector>

namespace hairsim {
struct RefConfig {
  std::vector<Vec3d> ref_pos;
  std::vector<Real> ref_theta;
  std::vector<Real> length;
  std::vector<Vec3d> m1;
  std::vector<Vec3d> m2;
  std::vector<Vec3d> e;
  std::vector<Vec4d> curv_vec;
  RefConfig(std::vector<Vec3d> ref_pos_, std::vector<Real> ref_theta_)
      : ref_pos(std::move(ref_pos_)), ref_theta(std::move(ref_theta_)) {
    int nVertices = ref_pos_.size();
    length.resize(nVertices - 1);
    m1.resize(nVertices - 1);
    m2.resize(nVertices - 1);
    e.resize(nVertices - 1);
    for (int i = 0; i < nVertices - 1; i++) {
      length[i] = (ref_pos[i + 1] - ref_pos[i]).norm();
      m1[i] = Vec3d(cos(ref_theta[i]), sin(ref_theta[i]), 0);
      m2[i] = Vec3d(-sin(ref_theta[i]), cos(ref_theta[i]), 0);
      e[i] = ref_pos[i + 1] - ref_pos[i];
      curv_vec.emplace_back(Vec4d(0, 0, 0, 0));
    }
  }
};

struct HairParams {
  Real E = 1e9;
  Real G = 3.65e8;
  Real radius = 5e-5;
  Vec3d color = {0.8, 0.8, 0.8};
  Real rho = 1e3;
  [[nodiscard]] Real area() const { return M_PI * radius * radius; }
};

struct System;

struct Hair : core::NonCopyable {
  // this is for initing the vertices on the hair by indices
  Hair(System* system_, int idx, RefConfig ref_config);
  Real& theta(int i) {
    assert(m_q.size() % 4 == 3);
    assert(4 * i + 3 < m_q.size());
    return m_q(4 * i + 3);
  }

  [[nodiscard]] Real theta(int i) const {
    assert(m_q.size() % 4 == 3);
    assert(4 * i + 3 < m_q.size());
    return m_q(4 * i + 3);
  }

  [[nodiscard]] Vec3d pos(int i) const {
    assert(m_q.size() % 4 == 3);
    assert(4 * i + 2 < m_q.size());
    return m_q.segment<3>(4 * i);
  }

  Eigen::Ref<Vec3d> pos(int i) {
    assert(m_q.size() % 4 == 3);
    assert(4 * i + 2 < m_q.size());
    return {m_q.segment<3>(4 * i)};
  }

  Eigen::Ref<Vec3d> vel(int i) {
    assert(m_qdot.size() % 4 == 3);
    assert(4 * i + 2 < m_qdot.size());
    return {m_qdot.segment<3>(4 * i)};
  }

  [[nodiscard]] Vec3d edge(Index i) const { return pos(i + 1) - pos(i); }

  [[nodiscard]] Vec3d tangent(Index i) const {
    return (pos(i + 1) - pos(i)).normalized();
  }

  Vec3d m1(Index i) const { return {}; }
  Vec3d m2(Index i) const { return {}; }

  Real twsitingAngle(Index i) const {
    return i > 0 ? theta(i) - theta(i - 1) : theta(i);
  }

  const Vec3d& curvatureBinormal(Index i) const { return m_kb[i]; }
  Real ShearModulus() const;
  Real YoungsModulus() const;

  [[nodiscard]] Real edgeLength(Index i) const {
    return (pos(i + 1) - pos(i)).norm();
  }

  [[nodiscard]] Real kappa1(Index i, Index j) const {
    return -m2(i).dot(curvatureBinormal(j));
  }

  [[nodiscard]] Real kappa2(Index i, Index j) const {
    return m1(i).dot(curvatureBinormal(j));
  }

  // (kappa1(i - 1, i), kappa1(i, i), -kappa2(i - 1, i), -kappa2(i, i))
  [[nodiscard]] const Vec4d& materialCurvature(Index i) const {
    return m_curv_vec[i];
  }

  [[nodiscard]] Vec3d tangentTilde(Index i) const {
    return (tangent(i - 1) + tangent(i)) /
           (1.0 + tangent(i - 1).dot(tangent(i)));
  }

  [[nodiscard]] Vec3d vertexTangent(Index i) const {
    return (tangent(i - 1) + tangent(i)) * 0.5;
  }

  [[nodiscard]] Real m(int i) const {
    assert(i >= 0);
    if (i > 0) return theta(i) - theta(i - 1) + ref.ref_theta[i];
    return theta(i);
  }

  Real vertexReferenceLength(Index i) const {
    assert(i >= 0);
    if (i > 0 && i < m_q.size() - 1)
      return
          0.5 * (ref.length[i] + ref.length[i - 1]);
    if (i == 0) return 0.5 * ref.length[0];
    return 0.5 * ref.length[m_q.size() - 2];
  }

  System* system;
  RefConfig ref;
  Eigen::Map<VecXd> m_q;
  Eigen::Map<VecXd> m_qdot;
  std::vector<Vec3d> m_a1;
  std::vector<Vec3d> m_a2;
  std::vector<Vec3d> m_kb;
  std::vector<Vec4d> m_curv_vec;
};
} // namespace hairsim

#endif // KISEKI_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_