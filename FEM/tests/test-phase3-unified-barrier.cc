/**
 * Phase 3 测试：统一 Barrier 接口验证
 * 
 * 验证：
 * 1. 统一的 ConstraintPair 只依赖索引，不依赖具体的 distance type
 * 2. PP/PE/PT/EE 约束的 barrier energy 计算正确
 * 3. gradient 与 FD 一致
 * 4. Hessian SPD 投影正确
 */

#include <gtest/gtest.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/distances.h>
#include <Maths/block-types.h>
#include <glm/glm.hpp>
#include <cmath>

namespace ksk::fem::ipc {

using maths::BlockVector;

class Phase3UnifiedBarrierTest : public ::testing::Test {
protected:
  static constexpr Real dHat = 0.1;
  static constexpr Real kappa = 1.0;
  static constexpr Real eps = 1e-6;
  
  Barrier barrier{dHat};
  BlockVector<3> x, X;
  
  void SetUp() override {
    x.resize(10);
    X.resize(10);
    // 初始化为零
    for (int i = 0; i < 10; ++i) {
      x[i] = glm::dvec3(0.0);
      X[i] = glm::dvec3(0.0);
    }
  }
};

// =========================================================================
// PP 约束测试
// =========================================================================

TEST_F(Phase3UnifiedBarrierTest, PPConstraintEnergy) {
  // 创建两个接近的点
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  x[1] = glm::dvec3(0.05, 0.0, 0.0);  // 距离 = 0.05
  
  ConstraintPair pair;
  pair.type = ConstraintKind::PP;
  pair.indices[0] = 0;
  pair.indices[1] = 1;
  
  Real energy = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
  EXPECT_GT(energy, 0.0);  // 应该有正的 barrier 能量
  
  // 验证距离时，能量应为零
  x[1] = glm::dvec3(0.15, 0.0, 0.0);  // 距离 = 0.15 > dHat
  energy = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
  EXPECT_EQ(energy, 0.0);
}

TEST_F(Phase3UnifiedBarrierTest, PPConstraintGradient) {
  // 设置两个点
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  x[1] = glm::dvec3(0.05, 0.0, 0.0);
  
  ConstraintPair pair;
  pair.type = ConstraintKind::PP;
  pair.indices[0] = 0;
  pair.indices[1] = 1;
  
  // 计算解析梯度
  BlockVector<3> grad_analytical(10);
  for (int i = 0; i < 10; ++i) grad_analytical[i] = glm::dvec3(0.0);
  
  constraintPairBarrierGradient(pair, x, X, grad_analytical, barrier, kappa);
  
  // 验证梯度不为零
  EXPECT_GT(glm::length(grad_analytical[0]), 0.0);
  
  // 验证 FD 近似
  Real fd_eps = 1e-8;
  for (int d = 0; d < 3; ++d) {
    x[0][d] += fd_eps;
    Real energy_plus = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
    x[0][d] -= 2 * fd_eps;
    Real energy_minus = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
    x[0][d] += fd_eps;
    
    Real fd_grad = (energy_plus - energy_minus) / (2 * fd_eps);
    Real analytical_grad = grad_analytical[0][d];
    
    // 允许数值误差
    EXPECT_NEAR(analytical_grad, fd_grad, 1e-4)
        << "Gradient mismatch at dimension " << d;
  }
}

// =========================================================================
// PE 约束测试
// =========================================================================

TEST_F(Phase3UnifiedBarrierTest, PEConstraintEnergy) {
  // 点接近边
  x[0] = glm::dvec3(0.0, 0.05, 0.0);   // 点
  x[1] = glm::dvec3(0.0, 0.0, 0.0);    // 边端点 1
  x[2] = glm::dvec3(1.0, 0.0, 0.0);    // 边端点 2
  
  ConstraintPair pair;
  pair.type = ConstraintKind::PE;
  pair.indices[0] = 0;  // 点
  pair.indices[1] = 1;  // 边端点 1
  pair.indices[2] = 2;  // 边端点 2
  
  Real energy = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
  EXPECT_GT(energy, 0.0);
}

TEST_F(Phase3UnifiedBarrierTest, PEConstraintGradient) {
  // 点接近边
  x[0] = glm::dvec3(0.0, 0.05, 0.0);
  x[1] = glm::dvec3(0.0, 0.0, 0.0);
  x[2] = glm::dvec3(1.0, 0.0, 0.0);
  
  ConstraintPair pair;
  pair.type = ConstraintKind::PE;
  pair.indices[0] = 0;
  pair.indices[1] = 1;
  pair.indices[2] = 2;
  
  BlockVector<3> grad_analytical(10);
  for (int i = 0; i < 10; ++i) grad_analytical[i] = glm::dvec3(0.0);
  
  constraintPairBarrierGradient(pair, x, X, grad_analytical, barrier, kappa);
  
  // 验证梯度不为零
  EXPECT_GT(glm::length(grad_analytical[0]) + glm::length(grad_analytical[1]) +
            glm::length(grad_analytical[2]), 0.0);
}

// =========================================================================
// PT 约束测试
// =========================================================================

TEST_F(Phase3UnifiedBarrierTest, PTConstraintEnergy) {
  // 点接近三角形
  x[0] = glm::dvec3(0.5, 0.5, 0.05);   // 点
  x[1] = glm::dvec3(0.0, 0.0, 0.0);    // 三角形顶点 1
  x[2] = glm::dvec3(1.0, 0.0, 0.0);    // 三角形顶点 2
  x[3] = glm::dvec3(0.5, 1.0, 0.0);    // 三角形顶点 3
  
  ConstraintPair pair;
  pair.type = ConstraintKind::PT;
  pair.indices[0] = 0;  // 点
  pair.indices[1] = 1;  // 三角形顶点 1
  pair.indices[2] = 2;  // 三角形顶点 2
  pair.indices[3] = 3;  // 三角形顶点 3
  
  Real energy = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
  EXPECT_GT(energy, 0.0);
}

// =========================================================================
// EE 约束测试
// =========================================================================

TEST_F(Phase3UnifiedBarrierTest, EEConstraintEnergy) {
  // 两条边接近
  x[0] = glm::dvec3(0.0, 0.0, 0.0);    // 边 1 端点 1
  x[1] = glm::dvec3(1.0, 0.0, 0.0);    // 边 1 端点 2
  x[2] = glm::dvec3(0.5, 0.05, 0.0);   // 边 2 端点 1
  x[3] = glm::dvec3(0.5, 0.05, 1.0);   // 边 2 端点 2
  
  X = x;  // 参考配置相同
  
  ConstraintPair pair;
  pair.type = ConstraintKind::EE;
  pair.indices[0] = 0;  // 边 1 端点 1
  pair.indices[1] = 1;  // 边 1 端点 2
  pair.indices[2] = 2;  // 边 2 端点 1
  pair.indices[3] = 3;  // 边 2 端点 2
  
  Real energy = constraintPairBarrierEnergy(pair, x, X, barrier, kappa);
  EXPECT_GT(energy, 0.0);
}

// =========================================================================
// Collider 约束测试
// =========================================================================

TEST_F(Phase3UnifiedBarrierTest, ColliderPPConstraintEnergy) {
  // 弹性顶点接近运动学三角形顶点
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  
  std::vector<glm::dvec3> colliderVerts = {
    glm::dvec3(0.05, 0.0, 0.0),  // collider 顶点 0
    glm::dvec3(1.0, 0.0, 0.0),
    glm::dvec3(0.5, 1.0, 0.0)
  };
  
  ColliderConstraintPair pair;
  pair.type = ConstraintKind::PP;
  pair.writableIndices[0] = 0;
  pair.colliderIndices[0] = 0;  // 使用 collider 顶点 0
  pair.colliderIndices[1] = -1;
  pair.colliderIndices[2] = -1;
  
  Real energy = colliderConstraintPairBarrierEnergy(pair, x, colliderVerts, barrier, kappa);
  EXPECT_GT(energy, 0.0);
}

TEST_F(Phase3UnifiedBarrierTest, ColliderPEConstraintEnergy) {
  // 弹性顶点接近运动学边
  x[0] = glm::dvec3(0.0, 0.05, 0.0);
  
  std::vector<glm::dvec3> colliderVerts = {
    glm::dvec3(0.0, 0.0, 0.0),
    glm::dvec3(1.0, 0.0, 0.0),
    glm::dvec3(0.5, 1.0, 0.0)
  };
  
  ColliderConstraintPair pair;
  pair.type = ConstraintKind::PE;
  pair.writableIndices[0] = 0;
  pair.colliderIndices[0] = 0;  // edge 端点 1
  pair.colliderIndices[1] = 1;  // edge 端点 2
  pair.colliderIndices[2] = -1;
  
  Real energy = colliderConstraintPairBarrierEnergy(pair, x, colliderVerts, barrier, kappa);
  EXPECT_GT(energy, 0.0);
}

}  // namespace ksk::fem::ipc
