//
// Created by creeper on 5/24/24.
//

#pragma once
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <Maths/equations.h>
#include <Maths/inclusion-root-finder.h>
#include <Spatify/lbvh.h>
#include <fem/system.h>
#include <fem/colliders.h>
#include <fem/trajectories.h>
#include <memory>
#include <span>

namespace ksk::fem::ipc {
constexpr Real kTCCD_EpsilonEE = 6.217248937900877e-15;
constexpr Real kTCCD_EpsilonVT = 6.661338147750939e-15;

using fem::System;

struct CCDQuery {
  glm::dvec3 x1{};
  glm::dvec3 x2{};
  glm::dvec3 x3{};
  glm::dvec3 x4{};
  glm::dvec3 u1{};
  glm::dvec3 u2{};
  glm::dvec3 u3{};
  glm::dvec3 u4{};
};

inline BBox<Real, 3>
computeTrajectoryBBox(const TriangleTrajectory &trajectory) {
  const auto &[x, p, toi, triangle] = trajectory;
  BBox<Real, 3> box;
  int verts[3] = {triangle.x, triangle.y, triangle.z};
  for (int i = 0; i < 3; i++) {
    const auto& pos = x[verts[i]];
    auto endPos = pos + p[verts[i]] * toi;
    box.expand({pos.x, pos.y, pos.z});
    box.expand({endPos.x, endPos.y, endPos.z});
  }
  return box;
}

// reserved for TCCD implementation
struct VertexTriangleQuery {
  const std::array<glm::dvec3, 4> &x;
  const std::array<glm::dvec3, 4> &p;
  glm::dvec3 operator()(Real t, Real a, Real b) const {
    return (1.0 - a - b) * (x[1] + p[1] * t) + a * (x[2] + p[2] * t) +
           b * (x[3] + p[3] * t) - x[0] - p[0] * t;
  }
  [[nodiscard]] maths::MultiInterval<3>
  cubeEpsilon(const maths::MultiInterval<3> &I) const {
    std::array<Real, 8> vx{};
    std::array<Real, 8> vy{};
    std::array<Real, 8> vz{};
    for (int i = 0; i < 8; i++) {
      int m = i & 1;
      int n = (i >> 1) & 1;
      int l = (i >> 2) & 1;
      Real tm = I[0].bounds[m];
      Real an = I[1].bounds[n];
      Real bl = I[2].bounds[l];
      auto v = this->operator()(tm, an, bl);
      vx[i] = v[0];
      vy[i] = v[1];
      vz[i] = v[2];
    }
    Real gamma_x = std::max(*std::ranges::max_element(vx), 1.0);
    Real gamma_y = std::max(*std::ranges::max_element(vy), 1.0);
    Real gamma_z = std::max(*std::ranges::max_element(vz), 1.0);
    Real ex = gamma_x * gamma_x * gamma_x * kTCCD_EpsilonVT;
    Real ey = gamma_y * gamma_y * gamma_y * kTCCD_EpsilonVT;
    Real ez = gamma_z * gamma_z * gamma_z * kTCCD_EpsilonVT;
    return {maths::Interval{.bounds = {-ex, ex}},
            maths::Interval{.bounds = {-ey, ey}},
            maths::Interval{.bounds = {-ez, ez}}};
  }
};

struct EdgeEdgeQuery {
  const std::array<glm::dvec3, 4> &x;
  const std::array<glm::dvec3, 4> &p;
  glm::dvec3 operator()(Real t, Real a, Real b) const {
    return (1.0 - a) * (x[0] + p[0] * t) + a * (x[1] + p[1] * t) +
           b * (x[2] + p[2] * t) + (1.0 - b) * (x[3] + p[3] * t);
  }
  [[nodiscard]] maths::MultiInterval<3>
  cubeEpsilon(const maths::MultiInterval<3> &I) const {
    std::array<Real, 8> vx{};
    std::array<Real, 8> vy{};
    std::array<Real, 8> vz{};
    for (int i = 0; i < 8; i++) {
      int m = i & 1;
      int n = (i >> 1) & 1;
      int l = (i >> 2) & 1;
      Real tm = I[0].bounds[m];
      Real an = I[1].bounds[n];
      Real bl = I[2].bounds[l];
      auto v = this->operator()(tm, an, bl);
      vx[i] = v[0];
      vy[i] = v[1];
      vz[i] = v[2];
    }
    Real gamma_x = std::max(*std::ranges::max_element(vx), 1.0);
    Real gamma_y = std::max(*std::ranges::max_element(vy), 1.0);
    Real gamma_z = std::max(*std::ranges::max_element(vz), 1.0);
    Real ex = gamma_x * gamma_x * gamma_x * kTCCD_EpsilonEE;
    Real ey = gamma_y * gamma_y * gamma_y * kTCCD_EpsilonEE;
    Real ez = gamma_z * gamma_z * gamma_z * kTCCD_EpsilonEE;
    return {maths::Interval{.bounds = {-ex, ex}},
            maths::Interval{.bounds = {-ey, ey}},
            maths::Interval{.bounds = {-ez, ez}}};
  }
};

struct Contact {
  glm::dvec3 pos{};
  Real t{1.0};
};

enum class CCDMode { EE, VT };

struct ACCDOptions {
  CCDMode mode{CCDMode::EE};
  const CCDQuery &query;
  Real toi{1.0};
  std::optional<Real> reservedDistance{};
};

maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi);
std::optional<Contact> eeCCD(const CCDQuery &query, Real toi);
std::optional<Contact> vtCCD(const CCDQuery &query, Real toi);

struct CollisionDetector {
  explicit CollisionDetector(const System& sys) : system(sys) {}
  
  std::optional<Real> detect(const maths::BlockVector<3> &p);
  std::optional<Real> runACCD(const ACCDOptions &options);
  
  [[nodiscard]] const spatify::LBVH<Real>& trianglesBVH() const {
    return triangles_bvh;
  }
  
  [[nodiscard]] const spatify::LBVH<Real>& edgesBVH() const {
    return edges_bvh;
  }
  
  void updateBVHs(const maths::BlockVector<3> &p, Real toi);

  // ─── 运动学体碰撞检测 ───
  void setKinematicBodies(const std::vector<Collider>* bodies) { m_kinBodies = bodies; }
  [[nodiscard]] std::optional<Real> detectDeformableVsKinematic(
      const maths::BlockVector<3>& p, Real dt);

private:
  std::optional<Real> detectVertexTriangleCollision(const maths::BlockVector<3> &p);
  std::optional<Real> detectEdgeEdgeCollision(const maths::BlockVector<3> &p);
  std::optional<Real> runACCD(CCDMode mode, std::array<glm::dvec3, 4> &x,
                             std::array<glm::dvec3, 4> &p,
                             Real toi) const;
  std::optional<Real> runACCDReserved(CCDMode mode,
                                     std::array<glm::dvec3, 4> &x,
                                     std::array<glm::dvec3, 4> &p,
                                     Real toi, Real reservedDistance) const;
  
  const System &system;
  Real s{0.1};
  spatify::LBVH<Real> triangles_bvh{};
  spatify::LBVH<Real> edges_bvh{};

  // 运动学体相关
  const std::vector<Collider>* m_kinBodies = nullptr;
  struct KinematicBVH {
    spatify::LBVH<Real> bvh;
  };
  std::vector<KinematicBVH> m_kinTriBVHs;
  
  void updateKinematicBVHs();
};

} // namespace fem::ipc
