//
// Created by creeper on 10/22/24.
//

#pragma once

#include <Core/deserializer.h>
#include <Core/reflection.h>
#include <Deform/strain-energy-density.h>
#include <Maths/block-sparse-matrix.h>
#include <memory>
#include "tet-mesh.h"

namespace ksk::fem {
struct ElasticTetMeshConfig {
  TetMesh mesh{};
  Real density{};
  REFLECT(mesh, density)
};

struct ElasticTetMesh {
  TetMesh mesh{};
  std::vector<Real> tetRefVolumes{};
  std::vector<deform::DeformationGradient<Real, 3>> tetDeformationGradients{};
  std::unique_ptr<deform::StrainEnergyDensity<Real>> energy{};
  Real density{};

  ElasticTetMesh() = default;
  
  ElasticTetMesh(TetMesh mesh,
                 std::unique_ptr<deform::StrainEnergyDensity<Real>> energy, 
                 Real density)
    : energy(std::move(energy)), density(density) {
    setMesh(std::move(mesh));
  }

  void init(const SubVector<Real>& x, const SubVector<Real>& xdot, const SubVector<Real>& X);
  [[nodiscard]] size_t dofDim() const {
    return m_numVertices * 3;
  }
  void updateDeformationEnergyGradient(SubVector<Real> x);
  void assembleEnergyGradient(const SubVector<Real>& primitiveGradSubView) const;
  [[nodiscard]] Real deformationEnergy() const;
  
  // New BlockSparseMatrix interfaces (Phase 2B)
  void assembleEnergyHessian(maths::BlockSparseMatrix<3> &globalH, int blockStart) const;
  void assembleMassMatrix(maths::BlockSparseMatrix<3> &globalMass, int blockStart) const;
  
  [[nodiscard]] std::span<const Triangle> getSurfaceView() const {
    return mesh.surfaceView();
  }
  [[nodiscard]] std::span<const Edge> getEdgesView() const {
    return mesh.surfaceEdgeView();
  }
  [[nodiscard]] size_t getVertexCount() const {
    return m_numVertices;
  }
  static ElasticTetMesh static_deserialize(const core::JsonNode& json);
  
 private:
  int m_numVertices{};

  void setMesh(TetMesh&& mesh) {
    this->mesh = std::move(mesh);
    m_numVertices = static_cast<int>(this->mesh.getVertices().size());
  }
  static void tetAssembleGlobal(SubVector<Real> grad, const Vector<Real, 12> &local, Tetrahedron tet) {
    for (int i = 0; i < 4; i++)
      grad.segment<3>(tet[i] * 3) += local.segment<3>(3 * i);
  }
  [[nodiscard]] size_t numTets() const {
    return mesh.tets.size();
  }
};
using SoftBody = ElasticTetMesh;
}

