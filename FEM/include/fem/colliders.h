#pragma once
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <fem/types.h>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace ksk::fem {

/// 运动剖面基类 — 描述运动学体的时变变换
struct MotionProfile {
  virtual ~MotionProfile() = default;
  
  /// 获取 t 时刻的变换矩阵 (world = T(t) * rest)
  [[nodiscard]] virtual glm::dmat4 transformAt(Real t) const = 0;
  
  /// 获取 t 时刻某局部坐标点的速度
  [[nodiscard]] virtual glm::dvec3 velocityAt(Real t, const glm::dvec3& localPos) const = 0;
};

/// 使用 std::function 的轻量运动实现
struct FunctionalMotion : MotionProfile {
  std::function<glm::dmat4(Real t)> transformFunc;
  std::function<glm::dvec3(Real, const glm::dvec3&)> velocityFunc;

  [[nodiscard]] glm::dmat4 transformAt(Real t) const override {
    return transformFunc ? transformFunc(t) : glm::dmat4(1.0);
  }

  [[nodiscard]] glm::dvec3 velocityAt(Real t, const glm::dvec3& localPos) const override {
    return velocityFunc ? velocityFunc(t, localPos) : glm::dvec3(0.0);
  }
};

// ─── 常用运动预设（工厂函数）───

/// 静止
inline std::unique_ptr<MotionProfile> staticMotion() {
  auto m = std::make_unique<FunctionalMotion>();
  m->transformFunc = [](Real) { return glm::dmat4(1.0); };
  m->velocityFunc = [](Real, const glm::dvec3&) { return glm::dvec3(0.0); };
  return m;
}

/// 匀速平动
inline std::unique_ptr<MotionProfile> constantVelocity(glm::dvec3 v) {
  auto m = std::make_unique<FunctionalMotion>();
  m->transformFunc = [v](Real t) { 
    return glm::translate(glm::dmat4(1.0), v * t); 
  };
  m->velocityFunc = [v](Real, const glm::dvec3&) { return v; };
  return m;
}

/// 匀角速度绕轴转动
inline std::unique_ptr<MotionProfile> constantRotation(glm::dvec3 axis, glm::dvec3 center, Real omega) {
  axis = glm::normalize(axis);
  auto m = std::make_unique<FunctionalMotion>();
  m->transformFunc = [=](Real t) {
    auto T = glm::translate(glm::dmat4(1.0), center);
    T = glm::rotate(T, omega * t, axis);
    return glm::translate(T, -center);
  };
  m->velocityFunc = [=](Real, const glm::dvec3& local) {
    return omega * glm::cross(axis, local - center);
  };
  return m;
}

/// 正弦往复振动
inline std::unique_ptr<MotionProfile> sinusoidalMotion(glm::dvec3 dir, Real amplitude, Real freq) {
  dir = glm::normalize(dir);
  Real w = 2.0 * glm::pi<Real>() * freq;
  auto m = std::make_unique<FunctionalMotion>();
  m->transformFunc = [=](Real t) {
    return glm::translate(glm::dmat4(1.0), dir * amplitude * std::sin(w * t));
  };
  m->velocityFunc = [=](Real t, const glm::dvec3&) {
    return dir * amplitude * w * std::cos(w * t);
  };
  return m;
}

// ─── 三角形网格类型定义 ───
struct TriangleMesh {
  std::vector<glm::dvec3> vertices;
  std::vector<glm::ivec3> triangles;
};

/// 运动学物体（即原设计中的 Collider — 不被求解驱动，但参与碰撞的几何体）
///
/// 碰撞几何用 variant 表示：可以是三角网格，也可以是解析 SDF。
/// 未来可考虑将 Collider 作为 KinematicBody 的 type alias。
struct Collider {
  struct MeshGeometry {
    std::shared_ptr<const TriangleMesh> mesh;
  };

  struct SDFGeometry {
    std::function<Real(const glm::dvec3& point)> signedDistance;
    std::function<glm::dvec3(const glm::dvec3& point)> gradient;  // ∇SDF, 指向外侧
  };

  using CollisionGeometry = std::variant<MeshGeometry, SDFGeometry>;
  CollisionGeometry geometry;
  
  std::unique_ptr<MotionProfile> motion;
  
  glm::dmat4 currentTransform{1.0};

  std::vector<glm::dvec3> currentVertices;
  
  void advanceTo(Real t) {
    currentTransform = motion->transformAt(t);
    if (auto* mg = std::get_if<MeshGeometry>(&geometry)) {
      const auto& rest = mg->mesh->vertices;
      currentVertices.resize(rest.size());
      for (size_t i = 0; i < rest.size(); i++)
        currentVertices[i] = glm::dvec3(currentTransform * glm::dvec4(rest[i], 1.0));
    }
  }
  
  [[nodiscard]] glm::dvec3 vertexVelocity(int idx, Real t) const {
    auto* mg = std::get_if<MeshGeometry>(&geometry);
    assert(mg && "vertexVelocity only valid for MeshGeometry");
    return motion->velocityAt(t, mg->mesh->vertices[idx]);
  }

  /// SDF 距离查询（将世界坐标逆变换到 rest 空间后调用 SDF）
  [[nodiscard]] Real sdfDistance(const glm::dvec3& worldPoint) const {
    auto* sdf = std::get_if<SDFGeometry>(&geometry);
    assert(sdf && "sdfDistance only valid for SDFGeometry");
    auto localPoint = glm::dvec3(glm::inverse(currentTransform) * glm::dvec4(worldPoint, 1.0));
    return sdf->signedDistance(localPoint);
  }

  [[nodiscard]] glm::dvec3 sdfGradient(const glm::dvec3& worldPoint) const {
    auto* sdf = std::get_if<SDFGeometry>(&geometry);
    assert(sdf && "sdfGradient only valid for SDFGeometry");
    auto localPoint = glm::dvec3(glm::inverse(currentTransform) * glm::dvec4(worldPoint, 1.0));
    // 梯度需要变换回世界空间（法线变换 = 逆转置的 3×3 部分）
    auto localGrad = sdf->gradient(localPoint);
    auto normalMat = glm::transpose(glm::inverse(glm::dmat3(currentTransform)));
    return glm::normalize(normalMat * localGrad);
  }
};

} // namespace ksk::fem
