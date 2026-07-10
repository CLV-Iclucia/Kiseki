#include <gtest/gtest.h>
#include <fem/fem-simulation.h>
#include <fem/constraints.h>
#include <Core/json.h>
#include <Maths/block-vector.h>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

using namespace ksk::fem;

// 辅助函数：创建一个简单的四面体网格用于测试
static std::vector<Primitive> createTetrahedron() {
  // 简单四面体：4个顶点，1个四面体单元
  std::vector<Primitive> primitives;
  Primitive prim;
  prim.type = Primitive::TET;
  prim.vertices = {0, 1, 2, 3};
  prim.youngsModulus = 1e6;
  prim.poissonsRatio = 0.3;
  prim.density = 1000.0;
  primitives.push_back(prim);
  return primitives;
}

// 辅助函数：创建四面体网格的顶点位置
static maths::BlockVector<3> createTetrahedronPositions() {
  maths::BlockVector<3> positions(4);
  // 简单四面体
  positions[0] = glm::dvec3(0.0, 0.0, 0.0);
  positions[1] = glm::dvec3(1.0, 0.0, 0.0);
  positions[2] = glm::dvec3(0.0, 1.0, 0.0);
  positions[3] = glm::dvec3(0.0, 0.0, 1.0);
  return positions;
}

// 测试 1: 正弦振动约束
TEST(PrescribedMotion, SinusoidalMotion) {
  ConstraintManager mgr;
  
  // 为顶点 3 添加正弦振动约束 (沿 y 轴)
  Real amplitude = 0.5;
  Real frequency = 1.0;
  Real w = 2.0 * glm::pi<Real>() * frequency;
  glm::dvec3 dir = glm::dvec3(0.0, 1.0, 0.0);
  
  // 初始位置
  maths::BlockVector<3> positions(4);
  positions[0] = glm::dvec3(0.0, 0.0, 0.0);
  positions[1] = glm::dvec3(1.0, 0.0, 0.0);
  positions[2] = glm::dvec3(0.0, 1.0, 0.0);
  positions[3] = glm::dvec3(0.0, 0.0, 1.0);
  
  // 添加约束：顶点 3 沿 y 轴正弦振动
  auto posFunc = [=](Real t) -> glm::dvec3 {
    return positions[3] + dir * amplitude * std::sin(w * t);
  };
  auto velFunc = [=](Real t) -> glm::dvec3 {
    return dir * amplitude * w * std::cos(w * t);
  };
  
  mgr.prescribeMotion(3, std::move(posFunc), std::move(velFunc));
  mgr.build(4);
  
  // 测试不同时间点的位置
  Real t = 0.0;
  auto target = mgr.allConstraints()[0].targetAt(t);
  EXPECT_NEAR(target.y, positions[3].y, 1e-10);  // sin(0) = 0
  
  t = 0.25 / frequency;  // T/4
  target = mgr.allConstraints()[0].targetAt(t);
  EXPECT_NEAR(target.y, positions[3].y + amplitude, 1e-6);  // sin(pi/2) = 1
  
  t = 0.5 / frequency;  // T/2
  target = mgr.allConstraints()[0].targetAt(t);
  EXPECT_NEAR(target.y, positions[3].y, 1e-6);  // sin(pi) = 0
  
  // 测试速度
  t = 0.0;
  auto vel = mgr.allConstraints()[0].targetVelocityAt(t);
  EXPECT_NEAR(vel.y, amplitude * w, 1e-6);  // cos(0) = 1
}

// 测试 2: enforcePosition 和 enforceVelocity
TEST(PrescribedMotion, EnforcePositionAndVelocity) {
  ConstraintManager mgr;
  
  Real amplitude = 0.1;
  Real frequency = 2.0;
  Real w = 2.0 * glm::pi<Real>() * frequency;
  
  auto posFunc = [=](Real t) -> glm::dvec3 {
    return glm::dvec3(0.0, amplitude * std::sin(w * t), 0.0);
  };
  auto velFunc = [=](Real t) -> glm::dvec3 {
    return glm::dvec3(0.0, amplitude * w * std::cos(w * t), 0.0);
  };
  
  mgr.prescribeMotion(0, std::move(posFunc), std::move(velFunc));
  mgr.build(1);
  
  // 测试 enforcePosition
  maths::BlockVector<3> x(1);
  x[0] = glm::dvec3(999.0, 999.0, 999.0);  // 初始值
  Real t = 0.0;
  mgr.enforcePosition(x, t);
  EXPECT_NEAR(x[0].x, 0.0, 1e-10);
  EXPECT_NEAR(x[0].y, 0.0, 1e-10);  // sin(0) = 0
  EXPECT_NEAR(x[0].z, 0.0, 1e-10);
  
  // 测试 enforceVelocity
  maths::BlockVector<3> xdot(1);
  xdot[0] = glm::dvec3(999.0, 999.0, 999.0);
  mgr.enforceVelocity(xdot, t);
  EXPECT_NEAR(xdot[0].x, 0.0, 1e-10);
  EXPECT_NEAR(xdot[0].y, amplitude * w, 1e-6);  // cos(0) = 1
  EXPECT_NEAR(xdot[0].z, 0.0, 1e-10);
}

// 测试 3: 多个顶点的时变约束
TEST(PrescribedMotion, MultipleVertices) {
  ConstraintManager mgr;
  
  // 顶点 0: 静止
  mgr.pinVertices({0}, createTetrahedronPositions());
  
  // 顶点 1: 正弦振动 (x 方向)
  auto posFunc1 = [](Real t) -> glm::dvec3 {
    return glm::dvec3(1.0 + 0.1 * std::sin(2.0 * glm::pi<Real>() * t), 0.0, 0.0);
  };
  auto velFunc1 = [](Real t) -> glm::dvec3 {
    return glm::dvec3(0.1 * 2.0 * glm::pi<Real>() * std::cos(2.0 * glm::pi<Real>() * t), 0.0, 0.0);
  };
  mgr.prescribeMotion(1, std::move(posFunc1), std::move(velFunc1));
  
  // 顶点 2: 正弦振动 (y 方向)
  auto posFunc2 = [](Real t) -> glm::dvec3 {
    return glm::dvec3(0.0, 1.0 + 0.2 * std::sin(2.0 * glm::pi<Real>() * 2.0 * t), 0.0);
  };
  auto velFunc2 = [](Real t) -> glm::dvec3 {
    return glm::dvec3(0.0, 0.2 * 2.0 * glm::pi<Real>() * 2.0 * std::cos(2.0 * glm::pi<Real>() * 2.0 * t), 0.0);
  };
  mgr.prescribeMotion(2, std::move(posFunc2), std::move(velFunc2));
  
  mgr.build(4);
  
  // 验证约束数量
  EXPECT_EQ(mgr.allConstraints().size(), 3);  // 1个pin + 2个prescribed
  
  // 验证 t=0 时的位置
  maths::BlockVector<3> x(4);
  mgr.enforcePosition(x, 0.0);
  EXPECT_NEAR(x[0].x, 0.0, 1e-10);  // pin
  EXPECT_NEAR(x[0].y, 0.0, 1e-10);
  EXPECT_NEAR(x[1].x, 1.0, 1e-10);  // sin(0) = 0
  EXPECT_NEAR(x[2].y, 1.0, 1e-10);  // sin(0) = 0
}

// 测试 4: 速度约束
TEST(VelocityConstraint, Basic) {
  ConstraintManager mgr;
  
  // 顶点 0: 恒定速度 (1, 0, 0)
  auto velFunc = [](Real) -> glm::dvec3 {
    return glm::dvec3(1.0, 0.0, 0.0);
  };
  mgr.prescribeVelocity(0, std::move(velFunc));
  mgr.build(1);
  
  // 测试速度
  maths::BlockVector<3> xdot(1);
  xdot[0] = glm::dvec3(0.0, 0.0, 0.0);
  mgr.enforceVelocity(xdot, 0.0);
  EXPECT_NEAR(xdot[0].x, 1.0, 1e-10);
  EXPECT_NEAR(xdot[0].y, 0.0, 1e-10);
  EXPECT_NEAR(xdot[0].z, 0.0, 1e-10);
  
  // 速度约束不应该影响位置
  maths::BlockVector<3> x(1);
  x[0] = glm::dvec3(5.0, 5.0, 5.0);
  mgr.enforcePosition(x, 0.0);
  EXPECT_NEAR(x[0].x, 5.0, 1e-10);  // 不变
  EXPECT_NEAR(x[0].y, 5.0, 1e-10);
  EXPECT_NEAR(x[0].z, 5.0, 1e-10);
}

// 测试 5: projectToFreeSpace 与时变约束的交互
TEST(PrescribedMotion, ProjectToFreeSpace) {
  ConstraintManager mgr;
  
  // 顶点 0: pin (全约束)
  mgr.pinVertices({0}, createTetrahedronPositions());
  
  // 顶点 1: 部分分量约束 (仅 y)
  mgr.pinComponent(1, 1, 0.0);  // y 分量
  
  mgr.build(4);
  
  // 测试投影
  maths::BlockVector<3> v(4);
  v[0] = glm::dvec3(1.0, 2.0, 3.0);
  v[1] = glm::dvec3(4.0, 5.0, 6.0);
  v[2] = glm::dvec3(7.0, 8.0, 9.0);
  v[3] = glm::dvec3(10.0, 11.0, 12.0);
  
  mgr.projectToFreeSpace(v);
  
  // 顶点 0 全约束 → 全零
  EXPECT_NEAR(v[0].x, 0.0, 1e-10);
  EXPECT_NEAR(v[0].y, 0.0, 1e-10);
  EXPECT_NEAR(v[0].z, 0.0, 1e-10);
  
  // 顶点 1 仅 y 约束 → y = 0
  EXPECT_NEAR(v[1].x, 4.0, 1e-10);  // 不变
  EXPECT_NEAR(v[1].y, 0.0, 1e-10);  // 置零
  EXPECT_NEAR(v[1].z, 6.0, 1e-10);  // 不变
  
  // 顶点 2,3 无约束 → 不变
  EXPECT_NEAR(v[2].x, 7.0, 1e-10);
  EXPECT_NEAR(v[3].x, 10.0, 1e-10);
}
