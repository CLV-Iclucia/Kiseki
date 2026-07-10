//
// IPC Collision Pair 实现
//

#include <fem/ipc/collision-pair.h>
#include <fem/ipc/constraint.h>
#include <cassert>

namespace ksk::fem::ipc {

// =========================================================================
// VertexTriangleCollisionPair
// =========================================================================

void VertexTriangleCollisionPair::updateDistanceState() {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  type = decidePointTriangleDistanceType(p, a, b, c);
}

Real VertexTriangleCollisionPair::distanceSqr() const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  switch (type) {
    case PointTriangleDistanceType::P_A:   return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:   return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:   return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:  return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:  return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:  return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC: return distanceSqrPointPlane(p, a, b, c);
    default: throw std::runtime_error("Unknown distance type in VT collision pair");
  }
}

void VertexTriangleCollisionPair::appendConstraintPair(ConstraintPairSet& out, Real dHatSqr) const {
  if (!isActive(dHatSqr)) return;
  
  ConstraintPair cp;
  
  // 根据 distance type 确定 constraint kind 并按规范填充 indices
  switch (type) {
    case PointTriangleDistanceType::P_A:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalVertex;
      cp.indices[1] = globalTriVerts[0];  // 顶点 A
      break;
    case PointTriangleDistanceType::P_B:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalVertex;
      cp.indices[1] = globalTriVerts[1];  // 顶点 B
      break;
    case PointTriangleDistanceType::P_C:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalVertex;
      cp.indices[1] = globalTriVerts[2];  // 顶点 C
      break;
    case PointTriangleDistanceType::P_AB:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalVertex;        // 点 P
      cp.indices[1] = globalTriVerts[0];   // 边顶点 0 (A)
      cp.indices[2] = globalTriVerts[1];   // 边顶点 1 (B)
      break;
    case PointTriangleDistanceType::P_BC:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalVertex;        // 点 P
      cp.indices[1] = globalTriVerts[1];   // 边顶点 0 (B)
      cp.indices[2] = globalTriVerts[2];   // 边顶点 1 (C)
      break;
    case PointTriangleDistanceType::P_CA:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalVertex;        // 点 P
      cp.indices[1] = globalTriVerts[2];   // 边顶点 0 (C)
      cp.indices[2] = globalTriVerts[0];   // 边顶点 1 (A)
      break;
    case PointTriangleDistanceType::P_ABC:
      cp.type = ConstraintKind::PT;
      cp.indices[0] = globalVertex;        // 点 P
      cp.indices[1] = globalTriVerts[0];   // 三角形顶点 0 (A)
      cp.indices[2] = globalTriVerts[1];   // 三角形顶点 1 (B)
      cp.indices[3] = globalTriVerts[2];   // 三角形顶点 2 (C)
      break;
    default:
      return;
  }
  
  out.pairs.push_back(cp);
}

// =========================================================================
// EdgeEdgeCollisionPair
// =========================================================================

void EdgeEdgeCollisionPair::updateDistanceState() {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;
  auto ea0 = x[globalEdgeA[0]];
  auto ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]];
  auto eb1 = x[globalEdgeB[1]];

  const glm::dvec3 u = ea1 - ea0;
  const glm::dvec3 v = eb1 - eb0;
  const glm::dvec3 w = ea0 - eb0;

  Real a = glm::dot(u, u);
  Real b = glm::dot(u, v);
  Real c = glm::dot(v, v);
  Real d = glm::dot(u, w);
  Real e = glm::dot(v, w);
  Real D = a * c - b * b;

  if (a == 0.0 && c == 0.0) { type = EdgeEdgeDistanceType::A_C; return; }
  else if (a == 0.0) { type = EdgeEdgeDistanceType::A_CD; return; }
  else if (c == 0.0) { type = EdgeEdgeDistanceType::AB_C; return; }

  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
    type = decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
    return;
  }

  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
  Real sN = (b * e - c * d);
  double tN, tD;
  if (sN <= 0.0) {
    tN = e; tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= D) {
    tN = e + b; tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = (a * e - b * d); tD = D;
    if (tN > 0.0 && tN < tD &&
        glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
      if (sN < D / 2) { tN = e; tD = c; default_case = EdgeEdgeDistanceType::A_CD; }
      else { tN = e + b; tD = c; default_case = EdgeEdgeDistanceType::B_CD; }
    }
  }

  if (tN <= 0.0) {
    if (-d <= 0.0) type = EdgeEdgeDistanceType::A_C;
    else if (-d >= a) type = EdgeEdgeDistanceType::B_C;
    else type = EdgeEdgeDistanceType::AB_C;
  } else if (tN >= tD) {
    if ((-d + b) <= 0.0) type = EdgeEdgeDistanceType::A_D;
    else if ((-d + b) >= a) type = EdgeEdgeDistanceType::B_D;
    else type = EdgeEdgeDistanceType::AB_D;
  } else {
    type = default_case;
  }
}

Real EdgeEdgeCollisionPair::distanceSqr() const {
  auto A = x[globalEdgeA[0]];
  auto B = x[globalEdgeA[1]];
  auto C = x[globalEdgeB[0]];
  auto D = x[globalEdgeB[1]];
  switch (type) {
    case EdgeEdgeDistanceType::A_C:   return distanceSqrPointPoint(A, C);
    case EdgeEdgeDistanceType::A_D:   return distanceSqrPointPoint(A, D);
    case EdgeEdgeDistanceType::B_C:   return distanceSqrPointPoint(B, C);
    case EdgeEdgeDistanceType::B_D:   return distanceSqrPointPoint(B, D);
    case EdgeEdgeDistanceType::AB_C:  return distanceSqrPointLine(A, B, C);
    case EdgeEdgeDistanceType::AB_D:  return distanceSqrPointLine(A, B, D);
    case EdgeEdgeDistanceType::A_CD:  return distanceSqrPointLine(A, C, D);
    case EdgeEdgeDistanceType::B_CD:  return distanceSqrPointLine(B, C, D);
    case EdgeEdgeDistanceType::AB_CD: return distanceSqrLineLine(A, B, C, D);
    default: throw std::runtime_error("Unknown distance type in EE collision pair");
  }
}

bool EdgeEdgeCollisionPair::usesMollifier() const {
  auto ea0 = x[globalEdgeA[0]], ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]], eb1 = x[globalEdgeB[1]];
  auto rest_ea0 = X[globalEdgeA[0]], rest_ea1 = X[globalEdgeA[1]];
  auto rest_eb0 = X[globalEdgeB[0]], rest_eb1 = X[globalEdgeB[1]];
  return gipc::needsMollifier(ea0, ea1, eb0, eb1, rest_ea0, rest_ea1, rest_eb0, rest_eb1);
}

void EdgeEdgeCollisionPair::appendConstraintPair(ConstraintPairSet& out, Real dHatSqr) const {
  if (!isActive(dHatSqr)) return;
  
  ConstraintPair cp;
  
  // 根据 distance type 确定 constraint kind 并按规范填充 indices
  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalEdgeA[0];  // 点 A (edge A 的端点 0)
      cp.indices[1] = globalEdgeB[0];  // 点 C (edge B 的端点 0)
      break;
    case EdgeEdgeDistanceType::A_D:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalEdgeA[0];  // 点 A (edge A 的端点 0)
      cp.indices[1] = globalEdgeB[1];  // 点 D (edge B 的端点 1)
      break;
    case EdgeEdgeDistanceType::B_C:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalEdgeA[1];  // 点 B (edge A 的端点 1)
      cp.indices[1] = globalEdgeB[0];  // 点 C (edge B 的端点 0)
      break;
    case EdgeEdgeDistanceType::B_D:
      cp.type = ConstraintKind::PP;
      cp.indices[0] = globalEdgeA[1];  // 点 B (edge A 的端点 1)
      cp.indices[1] = globalEdgeB[1];  // 点 D (edge B 的端点 1)
      break;
    case EdgeEdgeDistanceType::AB_C:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalEdgeB[0];   // 点 C (point P)
      cp.indices[1] = globalEdgeA[0];   // 边 A 顶点 0
      cp.indices[2] = globalEdgeA[1];   // 边 A 顶点 1
      break;
    case EdgeEdgeDistanceType::AB_D:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalEdgeB[1];   // 点 D (point P)
      cp.indices[1] = globalEdgeA[0];   // 边 A 顶点 0
      cp.indices[2] = globalEdgeA[1];   // 边 A 顶点 1
      break;
    case EdgeEdgeDistanceType::A_CD:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalEdgeA[0];   // 点 A (point P)
      cp.indices[1] = globalEdgeB[0];   // 边 B 顶点 0
      cp.indices[2] = globalEdgeB[1];   // 边 B 顶点 1
      break;
    case EdgeEdgeDistanceType::B_CD:
      cp.type = ConstraintKind::PE;
      cp.indices[0] = globalEdgeA[1];   // 点 B (point P)
      cp.indices[1] = globalEdgeB[0];   // 边 B 顶点 0
      cp.indices[2] = globalEdgeB[1];   // 边 B 顶点 1
      break;
    case EdgeEdgeDistanceType::AB_CD:
      cp.type = ConstraintKind::EE;
      cp.indices[0] = globalEdgeA[0];   // 边 A 顶点 0
      cp.indices[1] = globalEdgeA[1];   // 边 A 顶点 1
      cp.indices[2] = globalEdgeB[0];   // 边 B 顶点 0
      cp.indices[3] = globalEdgeB[1];   // 边 B 顶点 1
      break;
    default:
      return;
  }
  
  out.pairs.push_back(cp);
}

// =========================================================================
// ColliderVTCollisionPair
// =========================================================================

void ColliderVTCollisionPair::appendConstraintPair(ConstraintPairSet& out, Real dHatSqr) const {
  if (!isActive(dHatSqr)) return;
  
  ColliderConstraintPair ccp;
  ccp.writableIndices[0] = deformableVertex;
  
  // 根据 distance type 确定 constraint kind 并按规范存储 collider 点索引
  // colliderIndices 存储的是局部 collider 三角形的 vertex indices (0, 1, 2)
  switch (type) {
    case PointTriangleDistanceType::P_A:
      ccp.type = ConstraintKind::PP;
      // 只用第一个 collider 点，其他设为 -1
      ccp.colliderIndices[0] = 0;  // ka
      ccp.colliderIndices[1] = -1;
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_B:
      ccp.type = ConstraintKind::PP;
      ccp.colliderIndices[0] = 1;  // kb
      ccp.colliderIndices[1] = -1;
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_C:
      ccp.type = ConstraintKind::PP;
      ccp.colliderIndices[0] = 2;  // kc
      ccp.colliderIndices[1] = -1;
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_AB:
      ccp.type = ConstraintKind::PE;
      ccp.colliderIndices[0] = 0;  // ka (edge 端点 0)
      ccp.colliderIndices[1] = 1;  // kb (edge 端点 1)
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_BC:
      ccp.type = ConstraintKind::PE;
      ccp.colliderIndices[0] = 1;  // kb (edge 端点 0)
      ccp.colliderIndices[1] = 2;  // kc (edge 端点 1)
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_CA:
      ccp.type = ConstraintKind::PE;
      ccp.colliderIndices[0] = 2;  // kc (edge 端点 0)
      ccp.colliderIndices[1] = 0;  // ka (edge 端点 1)
      ccp.colliderIndices[2] = -1;
      break;
    case PointTriangleDistanceType::P_ABC:
      ccp.type = ConstraintKind::PT;
      ccp.colliderIndices[0] = 0;  // ka
      ccp.colliderIndices[1] = 1;  // kb
      ccp.colliderIndices[2] = 2;  // kc
      break;
    default:
      return;
  }
  
  out.colliderPairs.push_back(ccp);
}

} // namespace ksk::fem::ipc
