//
// IPC Collision Pair — broad phase 候选层
//
// 表示 broad phase / candidate phase 产生的碰撞对，表示"可能接近，需要检查"
// 不直接代表 barrier 执行单元，而是激活筛选的来源
//

#pragma once
#include <fem/simplex.h>
#include <fem/ipc/distances.h>
#include <Maths/block-types.h>

namespace ksk::fem {
namespace ipc {

enum class CollisionPairKind {
  VT,        // Vertex-Triangle
  EE,        // Edge-Edge
  ColliderVT, // Collider Vertex-Triangle
};

/// Collision Pair：broad phase 候选，待激活筛选
/// 职责：
/// - 保存 broad phase 拓扑配对
/// - 在当前 x 下计算 distance type 和 distance
/// - 判断是否激活（distance < dHat）
/// - 如需激活，生成对应的 Constraint Pair

// =========================================================================
// VertexTriangleCollisionPair：VT 候选对
// =========================================================================

struct VertexTriangleCollisionPair {
  Triangle triangle;
  const maths::BlockVector<3>& x;  // 全局 BlockVector 引用
  int globalVertex;                 // 全局顶点 block 索引
  int globalTriVerts[3];           // triangle 三顶点的全局 block 索引
  PointTriangleDistanceType type{}; // 当前构型下的距离类型

  void updateDistanceState();
  [[nodiscard]] Real distanceSqr() const;
  
  [[nodiscard]] bool isActive(Real dHatSqr) const {
    return distanceSqr() < dHatSqr;
  }
  
  /// 根据当前距离类型和距离，生成激活的 Constraint Pair
  /// 如未激活，不生成任何东西
  void appendConstraintPair(class ConstraintPairSet& out, Real dHatSqr) const;
};

// =========================================================================
// EdgeEdgeCollisionPair：EE 候选对
// =========================================================================

struct EdgeEdgeCollisionPair {
  const maths::BlockVector<3>& x;
  const maths::BlockVector<3>& X;
  int globalEdgeA[2];
  int globalEdgeB[2];
  EdgeEdgeDistanceType type{};

  void updateDistanceState();
  [[nodiscard]] Real distanceSqr() const;
  
  [[nodiscard]] bool isActive(Real dHatSqr) const {
    return distanceSqr() < dHatSqr;
  }
  
  void appendConstraintPair(class ConstraintPairSet& out, Real dHatSqr) const;

private:
  [[nodiscard]] bool usesMollifier() const;
};

// =========================================================================
// ColliderVTCollisionPair：Collider VT 候选对（单侧）
// =========================================================================

struct ColliderVTCollisionPair {
  const maths::BlockVector<3>& x;  // 全局弹性体位置
  int deformableVertex;             // 弹性体顶点全局 block 索引
  glm::dvec3 ka, kb, kc;           // kinematic 三角形当前位置 (非 DOF)
  PointTriangleDistanceType type{}; // 当前构型下的距离类型

  void updateDistanceState() {
    auto p = x[deformableVertex];
    type = decidePointTriangleDistanceType(p, ka, kb, kc);
  }

  [[nodiscard]] Real distanceSqr() const {
    return distanceSqrPointTriangle(x[deformableVertex], ka, kb, kc);
  }

  [[nodiscard]] bool isActive(Real dHatSqr) const {
    return distanceSqr() < dHatSqr;
  }

  void appendConstraintPair(class ConstraintPairSet& out, Real dHatSqr) const;
};

// =========================================================================
// CollisionPairSet：broad phase 候选容器
// =========================================================================

struct CollisionPairSet {
  std::vector<VertexTriangleCollisionPair> vtPairs;
  std::vector<EdgeEdgeCollisionPair> eePairs;
  std::vector<ColliderVTCollisionPair> colliderVTPairs;

  void clear() {
    vtPairs.clear();
    eePairs.clear();
    colliderVTPairs.clear();
  }
  
  [[nodiscard]] size_t totalCount() const {
    return vtPairs.size() + eePairs.size() + colliderVTPairs.size();
  }
};

} // namespace ipc
} // namespace ksk::fem
