//
// Created by creeper on 5/25/24.
//

#pragma once
#include "external-force.h"
#include "geometry-manager.h"
#include <Core/zip.h>
#include <Deform/strain-energy-density.h>
#include <Maths/block-vector.h>
#include <Maths/block-sparse-matrix.h>
#include <Spatify/lbvh.h>
#include <fem/colliders.h>
#include <fem/constraints.h>
#include <fem/primitive.h>

namespace ksk {
namespace fem::ipc {
struct CollisionDetector;
}

namespace fem {
using deform::DeformationGradient;
using deform::StrainEnergyDensity;
using maths::vectorize;

// the only thing System do is dispatching tasks to primitives
// and gather the dof to solve the system globally
struct System {

  maths::BlockVector<3> x{}, X{}, xdot{}, energyGradient{};

  [[nodiscard]] Real meshLengthScale() const { return m_meshLengthScale; }

  [[nodiscard]] const maths::BlockSparseMatrix<3> &blockMass() const { return m_blockMass; }

  void spdProjectHessian(maths::BlockSparseMatrix<3> &blockH) const;

  void updateDeformationEnergy();

  void updateDeformationEnergyGradient();

  [[nodiscard]] Real deformationEnergy() const;

  void updateCurrentConfig(const maths::BlockVector<3> &x_nxt);

  [[nodiscard]] const maths::BlockVector<3> &deformationEnergyGradient() const { return energyGradient; }

  [[nodiscard]] size_t dof() const { return x.scalarSize(); }
  System &init();
  [[nodiscard]] Real kineticEnergy() const {
    maths::BlockVector<3> Mv(x.numBlocks());
    m_blockMass.apply(xdot, Mv);
    return 0.5 * xdot.dot(Mv);
  }

  [[nodiscard]] Real potentialEnergy() const { return deformationEnergy(); }

  [[nodiscard]] Real totalEnergy() const {
    return kineticEnergy() + potentialEnergy();
  }

  [[nodiscard]] Real gravitationalPotentialEnergy() const {
    maths::BlockVector<3> g_vec(x.numBlocks());
    for (int i = 0; i < x.numBlocks(); i++)
      g_vec[i] = m_gravity;
    // M * g_vec
    maths::BlockVector<3> Mg(x.numBlocks());
    m_blockMass.apply(g_vec, Mg);
    // Vg = -(M * g) · x = -Mg · x
    return -Mg.dot(x);
  }

  [[nodiscard]] const Primitive &primitive(int id) const { return prs[id]; }
  Primitive &primitive(int id) { return prs[id]; }

  [[nodiscard]] const std::vector<Primitive> &primitives() const { return prs; }

  [[nodiscard]] int numTriangles() const { return nTriangles; }
  [[nodiscard]] int numEdges() const { return nEdges; }
  [[nodiscard]] int numVertices() const { return nVertices; }

  [[nodiscard]] int globalTriangleToPrimitive(int globalIdx) const {
    return m_geometryManager.getTriangleRef(globalIdx).primitiveId;
  }

  [[nodiscard]] int globalEdgeToPrimitive(int globalIdx) const {
    return m_geometryManager.getEdgeRef(globalIdx).primitiveId;
  }

  [[nodiscard]] int globalVertexToPrimitive(int globalIdx) const;

  [[nodiscard]] Triangle getTriangleVertices(int globalIdx) const {
    return m_geometryManager.getGlobalTriangle(globalIdx);
  }

  [[nodiscard]] Edge getGlobalEdge(int globalIdx) const {
    return m_geometryManager.getGlobalEdge(globalIdx);
  }

  [[nodiscard]] bool triangleContainsVertex(int triangleIdx,
                                            int vertexIdx) const {
    return m_geometryManager.triangleContainsVertex(triangleIdx, vertexIdx);
  }

  [[nodiscard]] bool checkEdgeAdjacent(int edgeA, int edgeB) const {
    return m_geometryManager.checkEdgeAdjacent(edgeA, edgeB);
  }

  [[nodiscard]] maths::BlockVector<3> computeAcceleration() const;

  // 弹性体管理 (Python bindings)
  void addPrimitive(Primitive&& p) { prs.push_back(std::move(p)); }

  // 约束管理
  ConstraintManager& constraints() { return m_constraints; }
  const ConstraintManager& constraints() const { return m_constraints; }
  
  // 重力设置
  void setGravity(glm::dvec3 g) { m_gravity = g; }
  [[nodiscard]] glm::dvec3 gravity() const { return m_gravity; }
  
  // 时间管理
  void advanceTime(Real dt) { m_currentTime += dt; }
  Real currentTime() const { return m_currentTime; }

  // 运动学体管理
  [[nodiscard]] std::vector<Collider>& colliders() { return m_colliders; }
  [[nodiscard]] const std::vector<Collider>& colliders() const { return m_colliders; }
  void advanceKinematicBodies(Real t) {
    for (auto& body : m_colliders)
      body.advanceTo(t);
  }

  template <typename Func> void sequentialDispatch(Func &&func) const {
    for (auto i = 0; i < prs.size(); i++)
      func(prs[i], i);
  }

  template <typename Func> void sequentialDispatch(Func &&func) {
    for (auto i = 0; i < prs.size(); i++)
      func(prs[i], i);
  }

  template <typename Func> void autoDispatch(Func &&func) {
      sequentialDispatch(func);
  }

  template <typename Func> void autoDispatch(Func &&func) const {
      sequentialDispatch(func);
  }

  void initGeometryManager();

  [[nodiscard]] const GeometryManager &geometryManager() const {
    return m_geometryManager;
  }

  friend maths::BlockVector<3> symbolicDeformationEnergyGradient(System &system);
  friend maths::BlockVector<3> numericalDeformationEnergyGradient(System &system);
  friend struct SystemBuilder;

private:
  void logSystemInfo() const;
  Real cachedEnergy{};
  bool use_parallel_dispatch{false};

  std::vector<Primitive> prs;
  std::vector<ExternalForce> externalForces;
  std::vector<int> dofStarts;
  GeometryManager m_geometryManager;

  int nTriangles{0};
  int nEdges{0};
  int nVertices{0};

  maths::BlockSparseMatrix<3> m_blockMass;
  Real m_meshLengthScale{std::numeric_limits<Real>::infinity()};

  ConstraintManager m_constraints;
  glm::dvec3 m_gravity{0.0, -9.81, 0.0};
  std::vector<Real> m_lumpedMass;      // 每顶点质量 (trace(M_ii)/3)
  Real m_currentTime{0.0};

  std::vector<Collider> m_colliders;

  void updateDeformationGradient();
  void buildBlockMassMatrix();
};

maths::BlockVector<3> symbolicDeformationEnergyGradient(System &system);
maths::BlockVector<3> numericalDeformationEnergyGradient(System &system);

struct SystemConfig {
  struct ConstraintConfig {
    std::string type;          // "pin" | "pin_component" | "prescribed_motion"
    std::vector<int> vertices{};
    int vertex{0};
    glm::bvec3 components{true, true, true};
    glm::dvec3 value{0.0};
    std::string component;      // "x", "y", "z" for pin_component
    // 用于 prescribed_motion
    struct MotionConfig {
      std::string type;        // "sinusoidal"
      glm::dvec3 axis{0.0};
      Real amplitude{0.0};
      Real frequency{0.0};
    };
    std::optional<MotionConfig> motion;
  };

  std::vector<Primitive> primitives{};
  std::vector<Collider> colliders{};
  std::vector<ExternalForce> externalForces{};
  std::vector<ConstraintConfig> constraints{};
  glm::dvec3 gravity{0.0, -9.81, 0.0};

  REFLECT(primitives, colliders)
};

struct SystemBuilder {
  System build(const core::JsonNode &json);
};
}
}
