//
// IPC Constraint Pair — GIPC barrier 接口
//
// 所有接触对使用 GIPC (Generalized IPC) barrier:
//   - VT: 根据 distance type 分派到 PP/PE/PT 的 GIPC PFPx
//   - EE: 根据 distance type 分派到 PP/PE/EE，近平行走 PEE mollifier
//
// GIPC 相比经典 IPC 的优势:
//   - Hessian 天然 SPD (无需额外投影)
//   - C³ 光滑 (RANK=2 barrier)
//   - 更好的 Newton 收敛性
//

#pragma once
#include <fem/ipc/distances.h>
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/gipc/pfpx.h>
#include <fem/ipc/gipc/hessian.h>
#include <fem/ipc/gipc/mollifier.h>
#include <fem/simplex.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-types.h>
#include <array>


namespace ksk::fem {
struct System;
namespace ipc {

using maths::LocalGrad;
using maths::LocalHessian;
using gipc::Barrier;

// =========================================================================
// ConstraintKind：统一的约束类型枚举
// =========================================================================

enum class ConstraintKind {
  PP = 0,  // Point-Point
  PE = 1,  // Point-Edge
  PT = 2,  // Point-Triangle
  EE = 3,  // Edge-Edge
};

// =========================================================================
// ConstraintPair：统一的普通双侧约束对
// =========================================================================

/// 统一的 active constraint pair
/// 
/// 索引布局规范（barrier 层只需看索引顺序，无需额外类型信息）：
/// 
/// - PP (Point-Point): indices[0..1]
///   - indices[0] = point A
///   - indices[1] = point B
///   无论来自 P-A/P-B/P-C/A-C/A-D/B-C/B-D，都统一为两点对
///
/// - PE (Point-Edge): indices[0..2]
///   - indices[0] = point P
///   - indices[1] = edge vertex 0
///   - indices[2] = edge vertex 1
///   无论来自 P-AB/P-BC/P-CA/AB-C/AB-D/A-CD/B-CD，都统一为点-边
///
/// - PT (Point-Triangle): indices[0..3]
///   - indices[0] = point P
///   - indices[1] = triangle vertex 0
///   - indices[2] = triangle vertex 1
///   - indices[3] = triangle vertex 2
///   只可能来自 P-ABC 或 P-ABC collider
///
/// - EE (Edge-Edge): indices[0..3]
///   - indices[0] = edge A vertex 0
///   - indices[1] = edge A vertex 1
///   - indices[2] = edge B vertex 0
///   - indices[3] = edge B vertex 1
///   只可能来自 AB-CD（近平行或相交），走 mollifier 路径
struct ConstraintPair {
  ConstraintKind type = ConstraintKind::PP;
  std::array<int, 4> indices = {-1, -1, -1, -1};  // 最多 4 个顶点 block 索引

  // 根据 type 推导实际使用的索引个数
  [[nodiscard]] int indexCount() const {
    switch (type) {
      case ConstraintKind::PP: return 2;
      case ConstraintKind::PE: return 3;
      case ConstraintKind::PT: return 4;
      case ConstraintKind::EE: return 4;
      default: return 0;
    }
  }
};

// =========================================================================
// ColliderConstraintPair：单侧 Collider 约束对
// =========================================================================

/// Collider 参与的 active constraint pair（单侧写回）
/// 类型和索引数由 type 推导：
/// - PP: writableIndices[0] (1 个可写顶点), colliderIndices[0..2] (3 个 collider 约束点)
/// - PE: writableIndices[0] (1 个可写顶点), colliderIndices[0..2] (3 个 collider 约束点)
/// - PT: writableIndices[0] (1 个可写顶点), colliderIndices[0..2] (3 个 collider 约束点)
struct ColliderConstraintPair {
  ConstraintKind type = ConstraintKind::PP;
  std::array<int, 1> writableIndices = {-1};      // 可写顶点 block 索引
  std::array<int, 3> colliderIndices = {-1, -1, -1}; // collider 三角形内的局部索引选择
  std::array<glm::dvec3, 3> colliderVertices = {};   // 可选：直接内嵌 collider 三角形当前几何
  bool hasEmbeddedColliderTriangle = false;
};

// =========================================================================
// ConstraintPairSet：active constraint 容器
// =========================================================================

struct ConstraintPairSet {
  // 普通双侧约束
  std::vector<ConstraintPair> pairs;
  std::array<int, 5> typeOffsets = {0, 0, 0, 0, 0};  // offsets for PP, PE, PT, EE, end
  
  // Collider 单侧约束
  std::vector<ColliderConstraintPair> colliderPairs;
  std::array<int, 4> colliderTypeOffsets = {0, 0, 0, 0};  // offsets for PP, PE, PT, end

  void clear() {
    pairs.clear();
    typeOffsets = {0, 0, 0, 0, 0};
    colliderPairs.clear();
    colliderTypeOffsets = {0, 0, 0, 0};
  }

  [[nodiscard]] size_t numConstraintPairs() const { return pairs.size(); }
  [[nodiscard]] size_t numColliderConstraintPairs() const { return colliderPairs.size(); }
  [[nodiscard]] size_t totalConstraints() const { return pairs.size() + colliderPairs.size(); }
  
  /// 获取普通约束中指定类型的区间 [start, end)
  [[nodiscard]] std::pair<int, int> getConstraintKindRange(ConstraintKind kind) const {
    int idx = static_cast<int>(kind);
    return {typeOffsets[idx], typeOffsets[idx + 1]};
  }
  
  /// 获取 Collider 约束中指定类型的区间 [start, end)
  [[nodiscard]] std::pair<int, int> getColliderConstraintKindRange(ConstraintKind kind) const {
    int idx = static_cast<int>(kind);
    return {colliderTypeOffsets[idx], colliderTypeOffsets[idx + 1]};
  }
};

// =========================================================================
// Phase 3: 统一的 Barrier 接口 — 仅依赖 ConstraintPair 的统一索引
// =========================================================================

/// 计算单个约束对的 barrier 能量
/// 根据 pair.type 分派到 PP/PE/PT/EE 的具体计算
Real constraintPairBarrierEnergy(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,  // 仅 EE mollifier 需要
    const Barrier& barrier,
    Real kappa);

/// 计算单个约束对的 barrier 梯度
void constraintPairBarrierGradient(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,  // 仅 EE mollifier 需要
    maths::BlockVector<3>& globalGradient,
    const Barrier& barrier,
    Real kappa);

/// 计算单个约束对的 barrier Hessian
void constraintPairBarrierHessian(
    const ConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const maths::BlockVector<3>& X,  // 仅 EE mollifier 需要
    maths::BlockSparseMatrix<3>& globalHessian,
    const Barrier& barrier,
    Real kappa);

/// Collider 约束对的 barrier 能量（单侧）
Real colliderConstraintPairBarrierEnergy(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    const Barrier& barrier,
    Real kappa);

/// Collider 约束对的 barrier 梯度（仅写入可写顶点 DOF）
void colliderConstraintPairBarrierGradient(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    maths::BlockVector<3>& globalGradient,
    const Barrier& barrier,
    Real kappa);

/// Collider 约束对的 barrier Hessian（仅写入可写顶点 DOF）
void colliderConstraintPairBarrierHessian(
    const ColliderConstraintPair& pair,
    const maths::BlockVector<3>& x,
    const std::vector<glm::dvec3>& colliderTriangleVertices,
    maths::BlockSparseMatrix<3>& globalHessian,
    const Barrier& barrier,
    Real kappa);

// =========================================================================
// 以下是旧的 barrier 计算接口，后续 Phase 3 迁移到统一结构
// 暂时保留以保证兼容性
// =========================================================================

struct VertexTriangleConstraint {
  Triangle triangle;
  const maths::BlockVector<3>& x;  // 全局 BlockVector 引用
  int globalVertex;                 // 全局顶点 block 索引
  int globalTriVerts[3];           // triangle 三顶点的全局 block 索引
  PointTriangleDistanceType type{};

  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;

  // GIPC barrier 接口
  [[nodiscard]] Real barrierEnergy(const Barrier& barrier, Real kappa) const;
  void assembleBarrierGradient(const Barrier& barrier,
                               maths::BlockVector<3>& globalGradient,
                               Real kappa) const;
  void assembleBarrierHessian(const Barrier& barrier,
                              maths::BlockSparseMatrix<3>& globalHessian,
                              Real kappa) const;
};

struct EdgeEdgeConstraint {
  const maths::BlockVector<3>& x;   // 全局当前配置
  const maths::BlockVector<3>& X;   // 全局参考配置 (mollifier 用)
  int globalEdgeA[2];               // edge A 两端点全局 block 索引
  int globalEdgeB[2];               // edge B 两端点全局 block 索引
  EdgeEdgeDistanceType type{};

  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;

  // GIPC barrier 接口
  [[nodiscard]] Real barrierEnergy(const Barrier& barrier, Real kappa) const;
  void assembleBarrierGradient(const Barrier& barrier,
                               maths::BlockVector<3>& globalGradient,
                               Real kappa) const;
  void assembleBarrierHessian(const Barrier& barrier,
                              maths::BlockSparseMatrix<3>& globalHessian,
                              Real kappa) const;

private:
  /// 是否走 mollifier 分支 (近平行 EE)
  [[nodiscard]] bool usesMollifier() const;
};

/// 弹性顶点 vs 运动学三角形的 barrier 约束 (单侧)
/// 运动学体的位置不计入 DOF，仅弹性体顶点有梯度/Hessian 贡献
struct DeformableKinematicVTConstraint {
  const maths::BlockVector<3>& x;  // 全局弹性体位置
  int deformableVertex;             // 弹性体顶点全局 block 索引
  glm::dvec3 ka, kb, kc;           // kinematic 三角形当前位置 (非 DOF)
  PointTriangleDistanceType type{};

  void updateDistanceType() {
    auto p = x[deformableVertex];
    type = decidePointTriangleDistanceType(p, ka, kb, kc);
  }

  [[nodiscard]] Real distanceSqr() const {
    return distanceSqrPointTriangle(x[deformableVertex], ka, kb, kc);
  }

  // GIPC barrier 接口 — 仅贡献弹性顶点的 DOF
  [[nodiscard]] Real barrierEnergy(const Barrier& barrier, Real kappa) const;
  void assembleBarrierGradient(const Barrier& barrier,
                               maths::BlockVector<3>& grad, Real kappa) const;
  void assembleBarrierHessian(const Barrier& barrier,
                              maths::BlockSparseMatrix<3>& H, Real kappa) const;
};

} // namespace ipc
} // namespace ksk::fem

