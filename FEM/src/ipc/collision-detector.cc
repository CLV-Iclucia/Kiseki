//
// Created by creeper on 5/25/24.
//

#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <Maths/equations.h>
#include <Maths/linalg-utils.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/distances.h>
#include <fem/system.h>
#include <tbb/tbb.h>
#include <spdlog/spdlog.h>

namespace ksk::fem::ipc {

// GLM-based mixed product: a · (b × c)
static inline Real mixedProduct(const glm::dvec3 &a,
                             const glm::dvec3 &b,
                             const glm::dvec3 &c) {
  return glm::dot(a, glm::cross(b, c));
}

using ksk::maths::CubicEquationRoots;

maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi) {
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  Real a = mixedProduct(u2_u1, u3_u1, u4_u1);
  Real b = mixedProduct(x2_x1, u3_u1, u4_u1) +
           mixedProduct(u2_u1, x3_x1, u4_u1) +
           mixedProduct(u2_u1, u3_u1, x4_x1);
  Real c = mixedProduct(u2_u1, x3_x1, x4_x1) +
           mixedProduct(x2_x1, u3_u1, x4_x1) +
           mixedProduct(x2_x1, x3_x1, u4_u1);
  Real d = mixedProduct(x2_x1, x3_x1, x4_x1);
  return maths::clampedCubicSolve({.a = a, .b = b, .c = c, .d = d}, 0.0, toi,
                                  1e-10);
}

std::optional<Contact> eeCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.numRoots)
    return std::nullopt;
  if (solution.infiniteSolutions) {
    // Degenerate case: all four points are always coplanar.
    // Return conservative nullopt instead of throwing — the ACCD path handles this safely.
    spdlog::debug("[CCD] eeCCD: infinite coplanar solutions detected, returning nullopt");
    return std::nullopt;
  }
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  std::optional<maths::BinaryLinearSolution> sol{};
  for (int i = 0; i < solution.numRoots; i++) {
    Real t = solution.roots[i];
    auto updated_x2_x1 = x2_x1 + u2_u1 * t;
    auto updated_x3_x1 = x3_x1 + u3_u1 * t;
    auto updated_x4_x1 = x4_x1 + u4_u1 * t;
    auto updated_x4_x3 = updated_x4_x1 - updated_x3_x1;
    bool hasSol = true;
    for (int j = 0; j < 3; j++) {
      int k = (j + 1) % 3;
      Real a00 = updated_x2_x1[j], a01 = -updated_x4_x3[j],
           a10 = updated_x2_x1[k], a11 = -updated_x4_x3[k];
      Real b0 = updated_x3_x1[j], b1 = -updated_x3_x1[k];
      auto linearSystem = maths::BinaryLinearSystem{
          .a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem))
        continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol)
        hasSol = false;
      break;
    }
    if (!hasSol)
      continue;
    [[unlikely]] if (!sol) {
      // the two edges are on the same line
      Real la = glm::length(updated_x2_x1);
      Real lb = glm::length(updated_x4_x3);
      auto updated_x4_x2 = updated_x4_x1 - updated_x2_x1;
      auto updated_x3_x2 = updated_x3_x1 - updated_x2_x1;
      Real longest_distance =
          std::max({la, lb, glm::length(updated_x4_x1),
                    glm::length(updated_x4_x2),
                    glm::length(updated_x3_x1),
                    glm::length(updated_x3_x2)});
      if (longest_distance > la + lb)
        continue;
      auto updated_x1 = x1 + u1 * t;
      auto updated_x2 = updated_x2_x1 + updated_x1;
      auto updated_x3 = updated_x3_x1 + updated_x1;
      if (longest_distance == la)
        return std::make_optional<Contact>(0.5 * updated_x4_x3 + updated_x3, t);
      if (longest_distance == lb)
        return std::make_optional<Contact>(0.5 * updated_x2_x1 + updated_x1, t);
      if (longest_distance == glm::length(updated_x4_x1))
        return std::make_optional<Contact>(0.5 * updated_x3_x2 + updated_x2, t);
      if (longest_distance == glm::length(updated_x4_x2))
        return std::make_optional<Contact>(0.5 * updated_x3_x1 + updated_x1, t);
      if (longest_distance == glm::length(updated_x3_x1))
        return std::make_optional<Contact>(0.5 * updated_x4_x2 + updated_x2, t);
      else
        return std::make_optional<Contact>(0.5 * updated_x4_x1 + updated_x1, t);
    }
    auto [a, b] = *sol;
    if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0)
      continue;
    auto pos = x1 + u1 * t + a * updated_x2_x1;
    return std::make_optional<Contact>(pos, t);
  }
  return std::nullopt;
}

std::optional<Contact> vtCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.numRoots)
    return std::nullopt;
  if (solution.infiniteSolutions) {
    // Degenerate case: vertex always on the triangle plane.
    // Return conservative nullopt — the ACCD path handles this safely.
    spdlog::debug("[CCD] vtCCD: infinite coplanar solutions detected, returning nullopt");
    return std::nullopt;
  }
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  std::optional<maths::BinaryLinearSolution> sol{};
  for (int i = 0; i < solution.numRoots; i++) {
    Real t = solution.roots[i];
    auto updated_x1 = x1 + u1 * t;
    auto updated_x2 = x2_x1 + u2_u1 * t + updated_x1;
    auto updated_x3 = x3_x1 + u3_u1 * t + updated_x1;
    auto updated_x4 = x4_x1 + u4_u1 * t + updated_x1;
    auto updated_x1_x4 = updated_x1 - updated_x4;
    auto updated_x2_x4 = updated_x2 - updated_x4;
    auto updated_x3_x4 = updated_x3 - updated_x4;
    bool hasSol = true;
    for (int j = 0; j < 3; j++) {
      int k = (j + 1) % 3;
      Real a00 = updated_x2_x4[j], a01 = updated_x3_x4[j],
           a10 = updated_x2_x4[k], a11 = updated_x3_x4[k];
      Real b0 = updated_x1_x4[j], b1 = updated_x1_x4[k];
      auto linearSystem = maths::BinaryLinearSystem{
          .a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem))
        continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol)
        hasSol = false;
      break;
    }
    if (!hasSol)
      continue;
    [[unlikely]] if (!sol)
      throw std::runtime_error("Infinite solutions detected in vtCCD for "
                               "barycentric coordinates, which is unexpected");

    const auto &[a, b] = *sol;
    if (a < 0.0 || b < 0.0 || a + b > 1.0)
      continue;
    return std::make_optional<Contact>(updated_x1, t);
  }
  return std::nullopt;
}

std::optional<Real> CollisionDetector::runACCDReserved(
    CCDMode mode, std::array<glm::dvec3, 4> &x,
    std::array<glm::dvec3, 4> &p, Real toi, Real reservedDistance) const {
  auto computeSquaredDistance = [&]() -> Real {
    if (mode == CCDMode::EE)
      return distanceSqrEdgeEdge(x[0], x[1], x[2], x[3]);
    return distanceSqrPointTriangle(x[0], x[1], x[2], x[3]);
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp =
        std::max(glm::length(p[0]), glm::length(p[1])) + std::max(glm::length(p[2]), glm::length(p[3]));
  else
    lp =
        glm::length(p[0]) + std::max({glm::length(p[1]), glm::length(p[2]), glm::length(p[3])});
  if (lp == 0.0)
    return std::nullopt;
  Real dSqr = computeSquaredDistance();
  Real g = s * (dSqr - reservedDistance * reservedDistance) /
           (std::sqrt(dSqr) + reservedDistance);
  Real t = 0.0;
  Real tl = (1.0 - s) * (dSqr - reservedDistance * reservedDistance) /
            ((std::sqrt(dSqr) + reservedDistance) * lp);
  constexpr int kMaxACCDIterations = 2000;
  int accdIter = 0;
  while (true) {
    for (int i = 0; i < 4; i++)
      x[i] += p[i] * tl;
    dSqr = computeSquaredDistance();
    if (t > 0.0 && (dSqr - reservedDistance * reservedDistance) /
                           (std::sqrt(dSqr) + reservedDistance) <
                       g + 1e-10)
      break;
    t += tl;
    if (t > toi)
      return std::nullopt;
    tl = 0.9 * (dSqr - reservedDistance * reservedDistance) /
         ((std::sqrt(dSqr) + reservedDistance) * lp);
    // Guard against infinite loop when tl approaches zero
    if (++accdIter >= kMaxACCDIterations) {
      spdlog::warn("[ACCD-Reserved] hit max iterations ({}), returning conservative toi={}", kMaxACCDIterations, t);
      return t > 0.0 ? std::optional<Real>(t) : std::nullopt;
    }
  }
  return t;
}

std::optional<Real>
CollisionDetector::runACCD(CCDMode mode, std::array<glm::dvec3, 4> &x,
                           std::array<glm::dvec3, 4> &p, Real toi) const {
  auto computeDistance = [&]() -> Real {
    if (mode == CCDMode::EE)
      return std::sqrt(distanceSqrEdgeEdge(x[0], x[1], x[2], x[3]));
    else
      return std::sqrt(distanceSqrPointTriangle(x[0], x[1], x[2], x[3]));
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp =
        std::max(glm::length(p[0]), glm::length(p[1])) + std::max(glm::length(p[2]), glm::length(p[3]));
  else
    lp =
        glm::length(p[0]) + std::max({glm::length(p[1]), glm::length(p[2]), glm::length(p[3])});
  if (lp == 0.0)
    return std::nullopt;
  Real dis = computeDistance();
  Real g = s * dis;
  Real t = 0.0;
  Real tl = (1.0 - s) * (dis / lp);
  constexpr int kMaxACCDIterations = 2000;
  int accdIter = 0;
  while (true) {
    for (int i = 0; i < 4; i++)
      x[i] += p[i] * tl;
    dis = computeDistance();
    if (dis < g + 1e-10) {
      [[unlikely]] if (t == 0.0)
        t = tl;
      break;
    }
    t += tl;
    if (t > toi)
      return std::nullopt;
    tl = 0.9 * dis / lp;
    // Guard against infinite loop when tl approaches zero
    if (++accdIter >= kMaxACCDIterations) {
      spdlog::warn("[ACCD] hit max iterations ({}), returning conservative toi={}", kMaxACCDIterations, t);
      return t > 0.0 ? std::optional<Real>(t) : std::nullopt;
    }
  }
  return t;
}

std::optional<Real> CollisionDetector::runACCD(const ACCDOptions &options) {
  const auto &[mode, query, toi, reservedDistance] = options;
  auto pBar = (query.u1 + query.u2 + query.u3 + query.u4) * 0.25;
  std::array<glm::dvec3, 4> x{query.x1, query.x2, query.x3, query.x4};
  std::array<glm::dvec3, 4> p{query.u1 - pBar, query.u2 - pBar,
                                     query.u3 - pBar, query.u4 - pBar};
  if (reservedDistance)
    return runACCDReserved(mode, x, p, toi, reservedDistance.value());
  return runACCD(mode, x, p, toi);
}

std::optional<Real>
CollisionDetector::detectVertexTriangleCollision(const maths::BlockVector<3> &p) {
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  
  tbb::parallel_for(0, system.numVertices(), [&](int vertexIdx) {
    auto vertexTrajectoryBBox = system.geometryManager().getTrajectoryAccessor(
        system.x, p, toi).vertexBBox(vertexIdx);
    
    trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system.triangleContainsVertex(triangleIdx, vertexIdx))
            return false;

          auto triangleVertices = system.getTriangleVertices(triangleIdx);
          
          CCDQuery query{
              .x1 = system.x[vertexIdx],
              .x2 = system.x[triangleVertices.x],
              .x3 = system.x[triangleVertices.y],
              .x4 = system.x[triangleVertices.z],
              .u1 = p[vertexIdx],
              .u2 = p[triangleVertices.x],
              .u3 = p[triangleVertices.y],
              .u4 = p[triangleVertices.z]
          };
          
          auto contact = runACCD(ACCDOptions{CCDMode::VT, query, toi});
          if (!contact)
            return false;
          
          if (*contact < toi) {
            hasContact.store(true);
            toi.store(*contact);
          }
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });
  
  if (hasContact)
    return toi;
  return std::nullopt;
}

std::optional<Real>
CollisionDetector::detectEdgeEdgeCollision(const maths::BlockVector<3> &p) {
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  
  tbb::parallel_for(0, system.numEdges(), [&](int edgeIdx) {
    auto edgeTrajectoryBBox = system.geometryManager().getTrajectoryAccessor(
        system.x, p, toi).edgeBBox(edgeIdx);
    
    edgesBVH().runSpatialQuery(
        [&](int otherEdgeIdx) -> bool {
          if (system.checkEdgeAdjacent(edgeIdx, otherEdgeIdx))
            return false;

          auto edgeVertices = system.getGlobalEdge(edgeIdx);
          auto otherEdgeVertices = system.getGlobalEdge(otherEdgeIdx);
          
          CCDQuery query{
              .x1 = system.x[edgeVertices[0]],
              .x2 = system.x[edgeVertices[1]],
              .x3 = system.x[otherEdgeVertices[0]],
              .x4 = system.x[otherEdgeVertices[1]],
              .u1 = p[edgeVertices[0]],
              .u2 = p[edgeVertices[1]],
              .u3 = p[otherEdgeVertices[0]],
              .u4 = p[otherEdgeVertices[1]]
          };
          
          auto contact = runACCD(ACCDOptions{CCDMode::EE, query, toi});
          if (!contact)
            return false;

          if (*contact < toi) {
            hasContact.store(true);
            toi.store(*contact);
          }
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });
  
  if (hasContact)
    return toi;
  return std::nullopt;
}

std::optional<Real> CollisionDetector::detect(const maths::BlockVector<3> &p) {
  auto vtCollision = detectVertexTriangleCollision(p);
  auto eeCollision = detectEdgeEdgeCollision(p);
  
  if (!vtCollision && !eeCollision)
    return std::nullopt;

  if (!vtCollision)
    return eeCollision;

  if (!eeCollision)
    return vtCollision;

  return std::min(*vtCollision, *eeCollision);
}

void CollisionDetector::updateBVHs(const maths::BlockVector<3> &p, Real toi) {
  auto trajectoryAccessor = system.geometryManager().getTrajectoryAccessor(
      system.x, p, toi);
  
  if (system.numTriangles() > 0) {
    struct BVHAdapter {
      using CoordType = Real;
      const decltype(trajectoryAccessor)& accessor;
      
      explicit BVHAdapter(const decltype(trajectoryAccessor)& acc) : accessor(acc) {}
      
      [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
        return accessor.triangleBBox(idx);
      }
      
      [[nodiscard]] int size() const {
        return accessor.triangleSize();
      }
    };
    
    BVHAdapter adapter(trajectoryAccessor);
    triangles_bvh.update(adapter);
  }
  
  if (system.numEdges() > 0) {
    struct BVHAdapter {
      using CoordType = Real;
      const decltype(trajectoryAccessor)& accessor;
      
      explicit BVHAdapter(const decltype(trajectoryAccessor)& acc) : accessor(acc) {}
      
      [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
        return accessor.edgeBBox(idx);
      }
      
      [[nodiscard]] int size() const {
        return accessor.edgeSize();
      }
    };
    
    BVHAdapter adapter(trajectoryAccessor);
    edges_bvh.update(adapter);
  }
}

void CollisionDetector::updateKinematicBVHs() {
  if (!m_kinBodies) return;
  
  m_kinTriBVHs.resize(m_kinBodies->size());
  
  for (size_t bodyIdx = 0; bodyIdx < m_kinBodies->size(); bodyIdx++) {
    const auto& body = (*m_kinBodies)[bodyIdx];
    auto* mg = std::get_if<Collider::MeshGeometry>(&body.geometry);
    if (!mg) continue;  // SDF 碰撞体不需要 BVH
    
    const auto& triangles = mg->mesh->triangles;
    
    // 构建 BVH adapter
    struct KinematicBVHAdapter {
      using CoordType = Real;
      const std::vector<glm::dvec3>& vertices;
      const std::vector<glm::ivec3>& triangles;
      
      [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
        const auto& tri = triangles[idx];
        BBox<Real, 3> box;
        box.expand({vertices[tri.x].x, vertices[tri.x].y, vertices[tri.x].z});
        box.expand({vertices[tri.y].x, vertices[tri.y].y, vertices[tri.y].z});
        box.expand({vertices[tri.z].x, vertices[tri.z].y, vertices[tri.z].z});
        return box;
      }
      
      [[nodiscard]] int size() const {
        return static_cast<int>(triangles.size());
      }
    };
    
    KinematicBVHAdapter adapter{body.currentVertices, triangles};
    m_kinTriBVHs[bodyIdx].bvh.update(adapter);
  }
}

std::optional<Real> CollisionDetector::detectDeformableVsKinematic(
    const maths::BlockVector<3>& p, Real dt) {
  if (!m_kinBodies || m_kinBodies->empty()) return std::nullopt;
  
  // 更新运动学体的 BVH
  updateKinematicBVHs();
  
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  
  for (size_t bodyIdx = 0; bodyIdx < m_kinBodies->size(); bodyIdx++) {
    const auto& body = (*m_kinBodies)[bodyIdx];
    
    // 仅处理 MeshGeometry (SDF 碰撞在 barrier 层单独处理)
    auto* mg = std::get_if<Collider::MeshGeometry>(&body.geometry);
    if (!mg) continue;
    
    const auto& triangles = mg->mesh->triangles;
    
    // 如果没有 BVH，跳过
    if (bodyIdx >= m_kinTriBVHs.size()) continue;
    
    tbb::parallel_for(0, system.numVertices(), [&](int vertexIdx) {
      // 弹性体顶点轨迹 bbox
      auto startPos = system.x[vertexIdx];
      auto endPos = startPos + p[vertexIdx] * toi.load();
      BBox<Real, 3> vertexBBox;
      vertexBBox.expand({startPos.x, startPos.y, startPos.z});
      vertexBBox.expand({endPos.x, endPos.y, endPos.z});
      
      m_kinTriBVHs[bodyIdx].bvh.runSpatialQuery(
          [&](int triIdx) -> bool {
            const auto& tri = triangles[triIdx];
            
            // 构造 CCDQuery:
            // x1: 弹性体顶点位置
            // x2, x3, x4: 运动学三角形顶点位置
            // u1: 弹性体搜索方向
            // u2, u3, u4: 运动学顶点速度 * dt
            CCDQuery query{
              .x1 = system.x[vertexIdx],
              .x2 = body.currentVertices[tri.x],
              .x3 = body.currentVertices[tri.y],
              .x4 = body.currentVertices[tri.z],
              .u1 = p[vertexIdx],
              .u2 = body.vertexVelocity(tri.x, system.currentTime()) * dt,
              .u3 = body.vertexVelocity(tri.y, system.currentTime()) * dt,
              .u4 = body.vertexVelocity(tri.z, system.currentTime()) * dt,
            };
            
            auto contact = runACCD(ACCDOptions{CCDMode::VT, query, toi.load()});
            if (contact && *contact < toi.load()) {
              hasContact.store(true);
              toi.store(*contact);
            }
            return true;
          },
          [&](const BBox<Real, 3>& bbox) { return vertexBBox.overlap(bbox); }
      );
    });
  }
  
  return hasContact ? std::optional<Real>(toi.load()) : std::nullopt;
}

} // namespace fem::ipc
