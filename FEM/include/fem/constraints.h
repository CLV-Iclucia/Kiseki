#pragma once
#include <glm/glm.hpp>
#include <Maths/block-vector.h>
#include <fem/types.h>
#include <functional>
#include <variant>
#include <vector>
#include <optional>

namespace ksk::fem {

/// overloaded helper (C++17 pattern for std::visit)
template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
struct System;
struct FixedConstraint {
  glm::dvec3 targetPosition{0.0};
};

struct MotionConstraint {
  std::function<glm::dvec3(Real t)> positionFunc;
  std::function<glm::dvec3(Real t)> velocityFunc;
};

struct VelocityConstraint {
  std::function<glm::dvec3(Real t)> velocityFunc;
};

using ConstraintData = std::variant<FixedConstraint, MotionConstraint, VelocityConstraint>;

struct VertexConstraint {
  int globalBlockIdx;
  glm::bvec3 mask;
  ConstraintData data;

  [[nodiscard]] glm::dvec3 targetAt(Real t) const {
    return std::visit(overloaded{
      [](const FixedConstraint& c) { return c.targetPosition; },
      [t](const MotionConstraint& c) { return c.positionFunc(t); },
      [](const VelocityConstraint&) { return glm::dvec3(0.0); },
    }, data);
  }

  [[nodiscard]] glm::dvec3 targetVelocityAt(Real t) const {
    return std::visit(overloaded{
      [](const FixedConstraint&) { return glm::dvec3(0.0); },
      [t](const MotionConstraint& c) {
        if (c.velocityFunc) return c.velocityFunc(t);
        return glm::dvec3(0.0); 
      },
      [t](const VelocityConstraint& c) { return c.velocityFunc(t); },
    }, data);
  }

  [[nodiscard]] bool isPositionConstraint() const {
    return std::holds_alternative<FixedConstraint>(data)
        || std::holds_alternative<MotionConstraint>(data);
  }
  [[nodiscard]] bool isVelocityConstraint() const {
    return std::holds_alternative<VelocityConstraint>(data);
  }
};

/// 约束管理器
class ConstraintManager {
public:
  void addConstraint(const VertexConstraint& c) { m_constraints.push_back(c); }
  
  /// 从初始配置中提取固定约束（如 pin 底部顶点）
  void pinVertices(const std::vector<int>& vertexIndices, const maths::BlockVector<3>& positions);
  
  /// 约束某些顶点的某些分量
  void pinComponent(int vertexIdx, int component, Real value);
  
  /// 时变位移约束
  void prescribeMotion(int vertexIdx, std::function<glm::dvec3(Real)> positionFunc, 
                       std::optional<std::function<glm::dvec3(Real)>> velocityFunc = std::nullopt);
  
  /// 速度约束
  void prescribeVelocity(int vertexIdx, std::function<glm::dvec3(Real)> velocity);
  
  // --- 查询接口 ---
  
  /// 该顶点的第 d 个分量是否被约束
  [[nodiscard]] bool isConstrained(int blockIdx, int component) const {
    if (blockIdx < 0 || blockIdx >= static_cast<int>(m_constraintMask.size())) return false;
    return m_constraintMask[blockIdx][component];
  }
  
  /// 该顶点是否完全被约束（3个分量都约束了）
  [[nodiscard]] bool isFullyConstrained(int blockIdx) const {
    if (blockIdx < 0 || blockIdx >= static_cast<int>(m_constraintMask.size())) return false;
    auto m = m_constraintMask[blockIdx];
    return m.x && m.y && m.z;
  }
  
  /// 获取所有 free (未约束) 的 block 索引
  [[nodiscard]] const std::vector<int>& freeBlocks() const { return m_freeBlocks; }
  
  /// 获取所有 constrained 的 block 索引
  [[nodiscard]] const std::vector<int>& constrainedBlocks() const { return m_constrainedBlocks; }
  
  /// 获取所有约束（用于迭代）
  [[nodiscard]] const std::vector<VertexConstraint>& allConstraints() const { return m_constraints; }
  
  /// 在 t 时刻，将约束目标应用至System（强制设置 constrained DOFs）
  void enforcePosition(System& system, Real t) const;
  
  /// 在 t 时刻，将约束速度写入 xdot
  void enforceVelocity(System& system, Real t) const;
  
  /// 将方向向量中被约束的分量清零（投影到 free 子空间）
  void projectToFreeSpace(maths::BlockVector<3>& direction) const;
  
  /// 将梯度中被约束的分量清零
  void zeroConstrainedGradient(maths::BlockVector<3>& gradient) const {
    projectToFreeSpace(gradient);  // 语义相同
  }
  
  /// 初始化: 扫描约束列表，建立 free/constrained 索引
  void build(int totalBlocks);
  
private:
  std::vector<VertexConstraint> m_constraints;
  std::vector<int> m_freeBlocks;
  std::vector<int> m_constrainedBlocks;
  
  // 快速查询: blockIdx → 该 block 哪些分量被约束
  std::vector<glm::bvec3> m_constraintMask;  // size = totalBlocks, false=free
};

} // namespace ksk::fem
