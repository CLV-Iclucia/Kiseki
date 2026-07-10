//
// Created by creeper on 10/31/24.
//
#include <Core/profiler.h>
#include <Maths/tensor.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
namespace ksk::fem {
using maths::vectorize;

void ElasticTetMesh::init(const SubVector<Real>& x, const SubVector<Real>& xdot, const SubVector<Real>& X) {
  tetRefVolumes.reserve(mesh.tets.size());
  tetDeformationGradients.reserve(mesh.tets.size());

  const auto& restVerts = mesh.getVertices();          // rest shape → Dm_inverse
  const auto& initVerts = mesh.getInitialPositions();  // initial config (may differ if pre-deformed)

  for (int i = 0; i < numTets(); i++) {
    auto &tet = mesh.tets[i];

    // Rest-shape edge matrix → defines Dm_inverse (reference configuration)
    Matrix<Real, 3, 3> local_X;
    for (int j = 0; j < 3; j++)
      local_X.col(j) = restVerts[tet[j + 1]] - restVerts[tet[0]];

    Real volume = local_X.determinant() / 6.0;
    assert(volume > 0 && "Tet has non-positive volume after orientation fix — mesh is degenerate");
    tetRefVolumes.emplace_back(volume);
    tetDeformationGradients.emplace_back(local_X);  // stores Dm_inverse from rest shape

    // Initial deformed edge matrix → initial F = local_x * Dm_inverse
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) = initVerts[tet[j + 1]] - initVerts[tet[0]];
    tetDeformationGradients.back().updateCurrentConfig(local_x);
  }
  mesh.commit(x, xdot, X);
}

void ElasticTetMesh::updateDeformationEnergyGradient(SubVector<Real> x) {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) =
          x.segment<3>(mesh.tets[i][j + 1] * 3) - x.segment<3>(mesh.tets[i][0] * 3);
    dg.updateCurrentConfig(local_x);
  }
}
void ElasticTetMesh::assembleEnergyGradient(
    const SubVector<Real> &primitiveGradSubView) const {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    auto grad = energy->computeEnergyGradient(dg);
    auto p_F_p_x = dg.gradient();
    Vector<Real, 12> grad_x =
        p_F_p_x.transpose() * vectorize(grad) * tetRefVolumes[i];
    tetAssembleGlobal(primitiveGradSubView, grad_x, mesh.tets[i]);
  }
}

Real ElasticTetMesh::deformationEnergy() const {
  Real sum = 0.0;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    sum += energy->computeEnergyDensity(dg) * tetRefVolumes[i];
  }
  return sum;
}

// New BlockSparseMatrix interface (Phase 2B) — Parallelized (Phase A-1)
void ElasticTetMesh::assembleEnergyHessian(
    maths::BlockSparseMatrix<3> &globalH, int blockStart) const {
  SIM_PROFILE_SCOPE("assembleElasticHessian");
  const int nTets = static_cast<int>(numTets());

  // For small meshes, stay serial to avoid TBB overhead
  if (nTets < 64) {
    for (int i = 0; i < nTets; i++) {
      auto &dg = tetDeformationGradients[i];
      auto hessian_F = energy->filteredEnergyHessian(dg);
      auto p_F_p_x = dg.gradient();
      Eigen::Matrix<Real, 12, 12> hessian_x =
          p_F_p_x.transpose() * hessian_F * p_F_p_x * tetRefVolumes[i];
      std::array<int, 4> bi = {
          blockStart + mesh.tets[i][0],
          blockStart + mesh.tets[i][1],
          blockStart + mesh.tets[i][2],
          blockStart + mesh.tets[i][3]};
      globalH.assembleBlock<4>(hessian_x, bi);
    }
    return;
  }

  // Parallel path: per-thread local BlockSparseMatrix, merged after
  tbb::enumerable_thread_specific<maths::BlockSparseMatrix<3>> localH(
      [&]() {
        maths::BlockSparseMatrix<3> local(globalH.blockRows(), globalH.blockCols());
        // Estimate: each tet contributes 16 blocks (4x4 tet connectivity)
        local.reserve(nTets * 16 / 4);
        return local;
      });

  tbb::parallel_for(0, nTets, [&](int i) {
    auto &dg = tetDeformationGradients[i];
    auto hessian_F = energy->filteredEnergyHessian(dg);
    auto p_F_p_x = dg.gradient();
    Eigen::Matrix<Real, 12, 12> hessian_x =
        p_F_p_x.transpose() * hessian_F * p_F_p_x * tetRefVolumes[i];
    std::array<int, 4> bi = {
        blockStart + mesh.tets[i][0],
        blockStart + mesh.tets[i][1],
        blockStart + mesh.tets[i][2],
        blockStart + mesh.tets[i][3]};
    localH.local().assembleBlock<4>(hessian_x, bi);
  });

  // Merge thread-local results into globalH
  for (auto &local : localH)
    globalH.addFrom(local);
}

void ElasticTetMesh::assembleMassMatrix(
    maths::BlockSparseMatrix<3> &globalMass, int blockStart) const {
  for (int i = 0; i < numTets(); i++) {
    Real volume = tetRefVolumes[i];
    std::array<int, 4> bi = {
        blockStart + mesh.tets[i][0],
        blockStart + mesh.tets[i][1],
        blockStart + mesh.tets[i][2],
        blockStart + mesh.tets[i][3]};
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++) {
        Real coeff = density * volume * (j == k ? 0.1 : 0.05);
        globalMass.addBlock(bi[j], bi[k], glm::dmat3(coeff));
      }
  }
}

ElasticTetMesh ElasticTetMesh::static_deserialize(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary for ElasticTetMesh");
  const auto &dict = json.as<core::JsonDict>();
  
  ElasticTetMesh result;
  
  if (dict.contains("mesh"))
    result.setMesh(TetMesh::static_deserialize(dict.at("mesh")));
  else
    throw std::runtime_error("ElasticTetMesh missing mesh field");

  if (dict.contains("density")) {
    result.density = dict.at("density").as<Real>();
  } else {
    throw std::runtime_error("ElasticTetMesh missing density field");
  }
  
  if (dict.contains("energy"))
    result.energy = deform::createStrainEnergyDensity<Real>(dict.at("energy"));
  else
    throw std::runtime_error("ElasticTetMesh missing energy field");

  return result;
}

} // namespace fem