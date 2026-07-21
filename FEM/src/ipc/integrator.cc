//
// Created by creeper on 10/25/24.
//
#include <Maths/linear-solver.h>
#include <Maths/block-solvers/block-pcg.h>
#include <Core/profiler.h>
#include <fem/integrator-factory.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/distances.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/integrator.h>
#include <fem/system.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>


namespace ksk::fem {

static int ipc_auto_reg = ([]() {
    IntegratorFactory::instance().registerCreator(
        "ipc",
        [](System &system, const core::JsonNode &json) {
          return IpcIntegrator::create(system, json);
        });
  }(), 0);

void IpcIntegrator::step(Real dt) {
  SIM_PROFILE_FUNCTION();

  Real t_next = system().currentTime() + dt;
  system().advanceKinematicBodies(t_next);

  collisionDetector->setKinematicBodies(&system().colliders());

  system().constraints().enforcePosition(system(), t_next);
  maths::BlockVector<3> x_t = system().x;
  x_prev = system().x;

  if (!hasInitializedActiveConstraints) {
    SIM_PROFILE_SCOPE("InitCollisionPairs");
    maths::BlockVector<3> zeroDirection(x_t.numBlocks());
    zeroDirection.setZero();
    precomputeCollisionPairs(zeroDirection, 0.0);
    refreshActiveConstraintPairs();
    hasInitializedActiveConstraints = true;
  }

  Real h = dt;

  Real E_prev;
  {
    SIM_PROFILE_SCOPE_COLOR("InitialEnergy", ksk::core::profiler_colors::kPurple);
    E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  }
  spdlog::info("[IPC] E_prev = {}", E_prev);
  maths::BlockVector<3> p(x_t.numBlocks());
  int iter = 0;
  while (true) {
    SIM_PROFILE_SCOPE("NewtonIteration");

    // Compute negative gradient directly as BlockVector<3>
    maths::BlockVector<3> negG;
    {
      SIM_PROFILE_SCOPE_COLOR("Gradient", ksk::core::profiler_colors::kRed);
      negG = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
      negG *= -1.0;
      system().constraints().zeroConstrainedGradient(negG);
    }

    // Compute Hessian directly as BlockSparseMatrix<3>
    maths::BlockSparseMatrix<3> H;
    {
      SIM_PROFILE_SCOPE_COLOR("HessianAssembly", ksk::core::profiler_colors::kCyan);
      H = spdProjectHessian(h);
    }

    // Linear solve
    {
      SIM_PROFILE_SCOPE_COLOR("LinearSolve", ksk::core::profiler_colors::kBlue);
      p.setZero();
      auto result = solver->solve(H, negG, p);
      if (!result.converged)
        spdlog::warn("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);
      else
        spdlog::info("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);
    }

    system().constraints().projectToFreeSpace(p);

    // Use scene bbox diagonal from LBVH root node as the length scale
    Real sceneScale = [&]() {
      const auto& bvh = collisionDetector->trianglesBVH();
      if (!bvh.bbox.empty())
        return glm::length(bvh.bbox[0].extent());
      return system().meshLengthScale();
    }();
    if (p.infNorm() <
        config.eps * sceneScale * h)
      break;

    // CCD + line search
    Real alpha;
    {
      SIM_PROFILE_SCOPE_COLOR("CCD", ksk::core::profiler_colors::kOrange);
      Real alphaElastic = computeStepSizeUpperBound(p);
      Real alphaKinematic = 1.0;
      if (!system().colliders().empty()) {
        if (auto toiColliders = collisionDetector->detectDeformableVsKinematic(p, dt)) alphaKinematic = *toiColliders;
      }
      alpha = config.stepSizeScale * std::min({1.0, alphaElastic, alphaKinematic});
    }
    spdlog::info("[IPC] iter {}: alpha={}", iter, alpha);

    if (alpha == 0.0)
      throw std::runtime_error(
          "Invalid state: collision happened within an integration step");

    {
      SIM_PROFILE_SCOPE_COLOR("LineSearch", ksk::core::profiler_colors::kGreen);
      precomputeCollisionPairs(p, alpha);
      Real E;
      do {
        maths::BlockVector<3> x_candidate = x_prev;
        x_candidate.axpy(alpha, p);
        updateCandidateSolution(x_candidate);
        system().constraints().enforcePosition(system(), t_next);
        E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
        if (E <= E_prev) break;
        alpha *= 0.5;
      } while (true);
      E_prev = E;
    }

    x_prev = system().x;
    iter++;
    SIM_PROFILE_VALUE("IPC/NewtonIter", iter);

    if (onNewtonIter)
      onNewtonIter(iter);
  }
  velocityUpdate(x_t, h);

  system().constraints().enforceVelocity(system(), t_next);

  system().advanceTime(dt);

  Real T = system().kineticEnergy();
  Real V = system().potentialEnergy();
  Real Vg = system().gravitationalPotentialEnergy();
  Real total = T + V + Vg;
  spdlog::info("[IPC] t={:.4f}  T={:.6e}  V_elastic={:.6e}  V_gravity={:.6e}  Total={:.6e}",
               system().currentTime(), T, V, Vg, total);

  SIM_PROFILE_VALUE("IPC/TotalEnergy", total);
  SIM_PROFILE_FRAME_MARK();
}

maths::BlockSparseMatrix<3> IpcIntegrator::spdProjectHessian(Real h) const
{
  SIM_PROFILE_SCOPE("spdProjectHessian");
  int nBlocks = system().x.numBlocks();
  maths::BlockSparseMatrix<3> H(nBlocks, nBlocks);

  // Elastic Hessian
  {
    SIM_PROFILE_SCOPE("spdProjectHessian/Elastic");
    system().spdProjectHessian(H);
  }

  Real kappa = config.contactStiffness;
  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  {
    SIM_PROFILE_SCOPE("spdProjectHessian/Barrier");
    for (int i = 0; i < activeConstraintPairCount; ++i) {
      const auto& pair = constraintPairs.pairs[i];
      ipc::constraintPairBarrierHessian(pair, system().x, system().X, H, barrier_, kappa);
    }
  }

  // H_total = h² * H_elastic_barrier + M
  {
    SIM_PROFILE_SCOPE("spdProjectHessian/PostProcess");
    H.scale(h * h);
    H.addFrom(system().blockMass());

    // Sort by row to build row-segment index, enabling parallel SpMV in BlockPCG.
    H.sortByRow();
  }

  return H;
}

void IpcIntegrator::refreshActiveConstraintPairs() {
  for (auto &c : collisionPairs.vtPairs)
    c.updateDistanceState();
  for (auto &c : collisionPairs.eePairs)
    c.updateDistanceState();
  for (auto &c : collisionPairs.colliderVTPairs)
    c.updateDistanceState();
  
  int ppCount = 0, peCount = 0, ptCount = 0, eeCount = 0;
  int colliderPpCount = 0, colliderPeCount = 0, colliderPtCount = 0;
  
  Real dHatSqr = barrier_.dHatSqr();
  
  for (const auto& cp : collisionPairs.vtPairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::PointTriangleDistanceType::P_A:
        case ipc::PointTriangleDistanceType::P_B:
        case ipc::PointTriangleDistanceType::P_C:
          ppCount++; break;
        case ipc::PointTriangleDistanceType::P_AB:
        case ipc::PointTriangleDistanceType::P_BC:
        case ipc::PointTriangleDistanceType::P_CA:
          peCount++; break;
        case ipc::PointTriangleDistanceType::P_ABC:
          ptCount++; break;
        default: break;
      }
    }
  }
  
  for (const auto& cp : collisionPairs.eePairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::EdgeEdgeDistanceType::A_C:
        case ipc::EdgeEdgeDistanceType::A_D:
        case ipc::EdgeEdgeDistanceType::B_C:
        case ipc::EdgeEdgeDistanceType::B_D:
          ppCount++; break;
        case ipc::EdgeEdgeDistanceType::AB_C:
        case ipc::EdgeEdgeDistanceType::AB_D:
        case ipc::EdgeEdgeDistanceType::A_CD:
        case ipc::EdgeEdgeDistanceType::B_CD:
          peCount++; break;
        case ipc::EdgeEdgeDistanceType::AB_CD:
          eeCount++; break;
        default: break;
      }
    }
  }
  
  for (const auto& cp : collisionPairs.colliderVTPairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::PointTriangleDistanceType::P_A:
        case ipc::PointTriangleDistanceType::P_B:
        case ipc::PointTriangleDistanceType::P_C:
          colliderPpCount++; break;
        case ipc::PointTriangleDistanceType::P_AB:
        case ipc::PointTriangleDistanceType::P_BC:
        case ipc::PointTriangleDistanceType::P_CA:
          colliderPeCount++; break;
        case ipc::PointTriangleDistanceType::P_ABC:
          colliderPtCount++; break;
        default: break;
      }
    }
  }
  
  int totalCount = ppCount + peCount + ptCount + eeCount;
  int totalColliderCount = colliderPpCount + colliderPeCount + colliderPtCount;
  
  if (constraintPairs.pairs.size() < totalCount)
    constraintPairs.pairs.resize(totalCount);
  if (constraintPairs.colliderPairs.size() < totalColliderCount)
    constraintPairs.colliderPairs.resize(totalColliderCount);
  
  int ppIdx = 0, peIdx = ppCount, ptIdx = ppCount + peCount, eeIdx = ppCount + peCount + ptCount;
  int colliderPpIdx = 0, colliderPeIdx = colliderPpCount, colliderPtIdx = colliderPpCount + colliderPeCount;
  
  for (const auto& cp : collisionPairs.vtPairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ConstraintPair c;
    switch (cp.type) {
      case ipc::PointTriangleDistanceType::P_A:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_B:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[2];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_AB:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        c.indices[2] = cp.globalTriVerts[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_BC:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[1];
        c.indices[2] = cp.globalTriVerts[2];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_CA:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[2];
        c.indices[2] = cp.globalTriVerts[0];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_ABC:
        c.type = ipc::ConstraintKind::PT;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        c.indices[2] = cp.globalTriVerts[1];
        c.indices[3] = cp.globalTriVerts[2];
        constraintPairs.pairs[ptIdx++] = c;
        break;
      default: break;
    }
  }
  
  for (const auto& cp : collisionPairs.eePairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ConstraintPair c;
    switch (cp.type) {
      case ipc::EdgeEdgeDistanceType::A_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::A_D:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_D:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_C:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeB[0];
        c.indices[1] = cp.globalEdgeA[0];
        c.indices[2] = cp.globalEdgeA[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_D:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeB[1];
        c.indices[1] = cp.globalEdgeA[0];
        c.indices[2] = cp.globalEdgeA[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::A_CD:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[0];
        c.indices[2] = cp.globalEdgeB[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_CD:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[0];
        c.indices[2] = cp.globalEdgeB[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_CD:
        c.type = ipc::ConstraintKind::EE;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeA[1];
        c.indices[2] = cp.globalEdgeB[0];
        c.indices[3] = cp.globalEdgeB[1];
        constraintPairs.pairs[eeIdx++] = c;
        break;
      default: break;
    }
    
  }
  
  for (const auto& cp : collisionPairs.colliderVTPairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ColliderConstraintPair c;
    c.writableIndices[0] = cp.deformableVertex;
    
    switch (cp.type) {
      case ipc::PointTriangleDistanceType::P_A:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_B:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 1;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_C:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 2;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_AB:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = 1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_BC:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 1;
        c.colliderIndices[1] = 2;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_CA:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 2;
        c.colliderIndices[1] = 0;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_ABC:
        c.type = ipc::ConstraintKind::PT;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = 1;
        c.colliderIndices[2] = 2;
        constraintPairs.colliderPairs[colliderPtIdx++] = c;
        break;
      default: break;
    }
    
  }
  
  constraintPairs.typeOffsets = {0, ppCount, ppCount + peCount, ppCount + peCount + ptCount, totalCount};
  constraintPairs.colliderTypeOffsets = {0, colliderPpCount, colliderPpCount + colliderPeCount, totalColliderCount};
  
 // constraintPairs.pairs.resize(totalCount);
 // constraintPairs.colliderPairs.resize(totalColliderCount);
}

void IpcIntegrator::precomputeCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  SIM_PROFILE_SCOPE("PrecomputeCollisionPairs");
  collisionPairs.clear();
  
  collisionDetector->updateBVHs(p, alpha);
  computeVertexTriangleCollisionPairs(p, alpha);
  computeEdgeEdgeCollisionPairs(p, alpha);
  
}

void IpcIntegrator::computeVertexTriangleCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  SIM_PROFILE_SCOPE("VT-CollisionPairs");
  Real dHat = config.dHat;
  const int nVerts = system().numVertices();

  tbb::enumerable_thread_specific<std::vector<ipc::VertexTriangleCollisionPair>> threadLocalVT;

  tbb::parallel_for(0, nVerts, [&](int vertexIdx) {
    auto vertexTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, alpha)
            .vertexBBox(vertexIdx);
    vertexTrajectoryBBox = vertexTrajectoryBBox.dilate(dHat);

    auto &local = threadLocalVT.local();
    collisionDetector->trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system().triangleContainsVertex(triangleIdx, vertexIdx))
            return false;

          auto globalTri = system().geometryManager().getGlobalTriangle(triangleIdx);
          local.push_back({
              .x = system().x,
              .globalVertex = vertexIdx,
              .globalTriVerts = {globalTri.x, globalTri.y, globalTri.z},
              .type = ipc::PointTriangleDistanceType::Unknown,
          });
          local.back().updateDistanceState();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });

  for (auto &local : threadLocalVT)
    for (auto &c : local)
      collisionPairs.vtPairs.push_back(c);
}

void IpcIntegrator::computeEdgeEdgeCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  SIM_PROFILE_SCOPE("EE-CollisionPairs");
  Real dHat = config.dHat;
  const int nEdges = system().numEdges();

  tbb::enumerable_thread_specific<std::vector<ipc::EdgeEdgeCollisionPair>> threadLocalEE;

  tbb::parallel_for(0, nEdges, [&](int edgeIdx) {
    auto edgeTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, alpha)
            .edgeBBox(edgeIdx);
    edgeTrajectoryBBox = edgeTrajectoryBBox.dilate(dHat);

    auto &local = threadLocalEE.local();
    collisionDetector->edgesBVH().runSpatialQuery(
        [&](int otherEdgeIdx) -> bool {
          if (system().checkEdgeAdjacent(edgeIdx, otherEdgeIdx))
            return false;

          auto globalEa = system().geometryManager().getGlobalEdge(edgeIdx);
          auto globalEb = system().geometryManager().getGlobalEdge(otherEdgeIdx);

          local.push_back({
              .x = system().x,
              .X = system().X,
              .globalEdgeA = {globalEa.x, globalEa.y},
              .globalEdgeB = {globalEb.x, globalEb.y},
              .type = ipc::EdgeEdgeDistanceType::Unknown,
          });
          local.back().updateDistanceState();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });

  for (auto &local : threadLocalEE)
    for (auto &c : local)
      collisionPairs.eePairs.push_back(c);
}

maths::BlockVector<3> IpcIntegrator::barrierEnergyGradient() const {
  maths::BlockVector<3> gradient(system().x.numBlocks());
  gradient.setZero();
  Real kappa = config.contactStiffness;

  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  for (int i = 0; i < activeConstraintPairCount; ++i) {
    const auto& pair = constraintPairs.pairs[i];
      ipc::constraintPairBarrierGradient(pair, system().x, system().X, gradient, barrier_, kappa);
  }

  return gradient;
}


std::unique_ptr<Integrator> IpcIntegrator::create(System &system,
                                                  const core::JsonNode &json) {
  std::unordered_map<std::string,
                     std::function<std::unique_ptr<IpcIntegrator>(
                         System &, const IpcIntegrator::Config &cfg)>>
      integratorCreators = {
          {"implicit-euler",
           [](System &system, const Config &cfg) {
             return std::make_unique<IpcImplicitEuler>(system, cfg);
           }},
      };

  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a JSON object for IpcIntegrator");
  const auto &dict = json.as<core::JsonDict>();
  if (!dict.contains("type"))
    throw std::runtime_error("IpcIntegrator missing type field");
  const auto &subtype = dict.at("type").as<std::string>();

  auto config = core::deserialize<Config>(dict.at("config"));
  auto integrator = integratorCreators.at(subtype)(system, config);

  // Create block solver (replaces legacy linearSolver creation)
  int maxIter = 1000;
  Real tol = 1e-6;
  if (dict.contains("solver")) {
    const auto &sDict = dict.at("solver").as<core::JsonDict>();
    if (sDict.contains("maxIterations")) maxIter = sDict.at("maxIterations").as<int>();
    if (sDict.contains("tolerance")) tol = sDict.at("tolerance").as<Real>();
  }
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(maxIter, tol);
  return integrator;
}

Real IpcIntegrator::barrierEnergy() const {
  Real energy = 0.0;
  Real kappa = config.contactStiffness;

  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  for (int i = 0; i < activeConstraintPairCount; ++i) {
    const auto& pair = constraintPairs.pairs[i];
    Real localEnergy = ipc::constraintPairBarrierEnergy(pair, system().x, system().X, barrier_, kappa);
    energy += localEnergy;
  }

  return energy;
}

} // namespace ksk::fem
