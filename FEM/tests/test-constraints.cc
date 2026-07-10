#include <gtest/gtest.h>
#include <fem/constraints.h>
#include <Maths/block-vector.h>

using namespace ksk::fem;

// 测试 1: projectToFreeSpace 单元测试
TEST(Constraints, ProjectToFreeSpace) {
  ConstraintManager mgr;
  
  // 添加约束: vertex 2, mask={true, false, true}
  VertexConstraint c;
  c.globalBlockIdx = 2;
  c.mask = {true, false, true};
  c.data = FixedConstraint{glm::dvec3(0.0, 0.0, 0.0)};
  mgr.addConstraint(c);
  
  // 构建管理器
  mgr.build(5);
  
  // 创建测试向量
  maths::BlockVector<3> v(5);
  for (int i = 0; i < 5; i++) {
    v[i] = glm::dvec3(i * 1.0, i * 2.0, i * 3.0);
  }
  
  // 投影到自由空间
  mgr.projectToFreeSpace(v);
  
  // 验证: vertex 2 的 x 和 z 分量应该被清零
  EXPECT_DOUBLE_EQ(v[2].x, 0.0);
  EXPECT_DOUBLE_EQ(v[2].y, 4.0);  // 未约束, 保持原值
  EXPECT_DOUBLE_EQ(v[2].z, 0.0);
  
  // 验证其他顶点未被修改
  EXPECT_DOUBLE_EQ(v[0].x, 0.0);
  EXPECT_DOUBLE_EQ(v[1].y, 2.0);
}

// 测试 2: enforcePosition 测试
TEST(Constraints, EnforcePosition) {
  ConstraintManager mgr;
  
  // 添加固定约束: vertex 0, 目标位置 (1, 2, 3)
  VertexConstraint c;
  c.globalBlockIdx = 0;
  c.mask = {true, true, true};
  c.data = FixedConstraint{glm::dvec3(1.0, 2.0, 3.0)};
  mgr.addConstraint(c);
  
  mgr.build(3);
  
  // 创建位置向量
  maths::BlockVector<3> x(3);
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  x[1] = glm::dvec3(4.0, 5.0, 6.0);
  x[2] = glm::dvec3(7.0, 8.0, 9.0);
  
  // 强制执行约束
  mgr.enforcePosition(x, 0.0);
  
  // 验证: vertex 0 应该被设置为目标位置
  EXPECT_DOUBLE_EQ(x[0].x, 1.0);
  EXPECT_DOUBLE_EQ(x[0].y, 2.0);
  EXPECT_DOUBLE_EQ(x[0].z, 3.0);
  
  // 验证其他顶点未被修改
  EXPECT_DOUBLE_EQ(x[1].x, 4.0);
  EXPECT_DOUBLE_EQ(x[2].z, 9.0);
}

// 测试 3: pinComponent 测试
TEST(Constraints, PinComponent) {
  ConstraintManager mgr;
  
  // 约束 vertex 1 的 y 分量
  mgr.pinComponent(1, 1, 5.0);  // component 1 = y
  
  mgr.build(3);
  
  // 验证约束掩码
  EXPECT_TRUE(mgr.isConstrained(1, 1));  // y 分量被约束
  EXPECT_FALSE(mgr.isConstrained(1, 0)); // x 分量未约束
  EXPECT_FALSE(mgr.isConstrained(1, 2)); // z 分量未约束
}

// 测试 4: 时变约束测试
TEST(Constraints, TimeVaryingConstraint) {
  ConstraintManager mgr;
  
  // 添加时变约束: vertex 0, 位置随时间和 sin 变化
  auto posFunc = [](Real t) -> glm::dvec3 {
    return glm::dvec3(std::sin(t), std::cos(t), t);
  };
  
  auto velFunc = [](Real t) -> glm::dvec3 {
    return glm::dvec3(std::cos(t), -std::sin(t), 1.0);
  };
  
  mgr.prescribeMotion(0, posFunc, velFunc);
  
  mgr.build(2);
  
  // 创建位置向量
  maths::BlockVector<3> x(2);
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  
  // 在 t=0 时强制执行约束
  mgr.enforcePosition(x, 0.0);
  
  // 验证: sin(0)=0, cos(0)=1, t=0
  EXPECT_NEAR(x[0].x, 0.0, 1e-10);
  EXPECT_NEAR(x[0].y, 1.0, 1e-10);
  EXPECT_NEAR(x[0].z, 0.0, 1e-10);
  
  // 在 t=π/2 时强制执行约束
  x[0] = glm::dvec3(0.0, 0.0, 0.0);
  mgr.enforcePosition(x, glm::pi<Real>() / 2.0);
  
  // 验证: sin(π/2)=1, cos(π/2)=0, t=π/2
  EXPECT_NEAR(x[0].x, 1.0, 1e-10);
  EXPECT_NEAR(x[0].y, 0.0, 1e-10);
  EXPECT_NEAR(x[0].z, glm::pi<Real>() / 2.0, 1e-10);
}

// 测试 5: 速度约束测试
TEST(Constraints, VelocityConstraint) {
  ConstraintManager mgr;
  
  // 添加速度约束: vertex 1, 恒定速度 (1, 0, -1)
  auto velFunc = [](Real) -> glm::dvec3 {
    return glm::dvec3(1.0, 0.0, -1.0);
  };
  
  mgr.prescribeVelocity(1, velFunc);
  
  mgr.build(3);
  
  // 创建速度向量
  maths::BlockVector<3> xdot(3);
  xdot[0] = glm::dvec3(0.0, 0.0, 0.0);
  xdot[1] = glm::dvec3(0.0, 0.0, 0.0);
  xdot[2] = glm::dvec3(0.0, 0.0, 0.0);
  
  // 强制执行速度约束
  mgr.enforceVelocity(xdot, 0.0);
  
  // 验证: vertex 1 的速度应该被设置为 (1, 0, -1)
  EXPECT_DOUBLE_EQ(xdot[1].x, 1.0);
  EXPECT_DOUBLE_EQ(xdot[1].y, 0.0);
  EXPECT_DOUBLE_EQ(xdot[1].z, -1.0);
  
  // 验证其他顶点未被修改
  EXPECT_DOUBLE_EQ(xdot[0].x, 0.0);
  EXPECT_DOUBLE_EQ(xdot[2].z, 0.0);
}

// 测试 6: 完全约束测试
TEST(Constraints, FullyConstrained) {
  ConstraintManager mgr;
  
  // 添加完全约束: vertex 0
  VertexConstraint c;
  c.globalBlockIdx = 0;
  c.mask = {true, true, true};
  c.data = FixedConstraint{glm::dvec3(0.0, 0.0, 0.0)};
  mgr.addConstraint(c);
  
  mgr.build(3);
  
  // 验证完全约束
  EXPECT_TRUE(mgr.isFullyConstrained(0));
  EXPECT_FALSE(mgr.isFullyConstrained(1));
  
  // 验证自由/约束块索引
  const auto& freeBlocks = mgr.freeBlocks();
  const auto& constrainedBlocks = mgr.constrainedBlocks();
  
  // vertex 0 应该在约束列表中
  EXPECT_TRUE(std::find(constrainedBlocks.begin(), constrainedBlocks.end(), 0) != constrainedBlocks.end());
  
  // vertex 1 和 2 应该在自由列表中
  EXPECT_TRUE(std::find(freeBlocks.begin(), freeBlocks.end(), 1) != freeBlocks.end());
  EXPECT_TRUE(std::find(freeBlocks.begin(), freeBlocks.end(), 2) != freeBlocks.end());
}
