#pragma once
#include <Renderer/scene-proxy.h>
#include <fem/system.h>
#include <fem/colliders.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <variant>

namespace sim::renderer {

/// 从 FEM System 构建渲染帧（公共工具函数）。
/// 遍历所有 Primitive + Collider，统一提取渲染数据。
/// 调用方只需一行：buildSceneProxyFromSystem(system, step)
inline std::unique_ptr<SceneProxy> buildSceneProxyFromSystem(
    const fem::System& system, int frameIndex) {
  auto proxy = std::make_unique<SceneProxy>();
  proxy->frameIndex = frameIndex;
  proxy->simulationTime = static_cast<float>(system.currentTime());

  // ─── Deformable bodies ───
  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& pr = system.primitives()[i];
    auto surfaceView = pr.getSurfaceView();
    auto vertCount = pr.getVertexCount();
    int dofStart = pr.getDofStart();

    MeshProxy mesh;
    mesh.name = "primitive_" + std::to_string(i);

    // double → float 位置转换
    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      const auto& pos = system.x[(dofStart / 3) + static_cast<int>(v)];
      mesh.positions[v] = glm::vec3(pos); // dvec3 → vec3
    }

    // 三角形索引（primitive 内部局部索引）
    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = {
          static_cast<unsigned>(tri.x),
          static_cast<unsigned>(tri.y),
          static_cast<unsigned>(tri.z)};
    }

    computeSmoothNormals(mesh);
    proxy->meshes.push_back(std::move(mesh));
  }

  // ─── Kinematic colliders (mesh geometry only) ───
  for (int ci = 0; ci < static_cast<int>(system.colliders().size()); ci++) {
    const auto& collider = system.colliders()[ci];
    auto* mg = std::get_if<fem::Collider::MeshGeometry>(&collider.geometry);
    if (!mg) continue;

    MeshProxy mesh;
    mesh.name = "collider_" + std::to_string(ci);
    mesh.objectColor = {0.6f, 0.6f, 0.6f};  // 灰色，区分于弹性体

    // 使用 currentVertices（已经过运动变换的世界坐标）
    const auto& verts = collider.currentVertices;
    mesh.positions.resize(verts.size());
    for (size_t v = 0; v < verts.size(); v++)
      mesh.positions[v] = glm::vec3(verts[v]);  // dvec3 → vec3

    const auto& tris = mg->mesh->triangles;
    mesh.triangles.resize(tris.size());
    for (size_t t = 0; t < tris.size(); t++)
      mesh.triangles[t] = {
          static_cast<unsigned>(tris[t].x),
          static_cast<unsigned>(tris[t].y),
          static_cast<unsigned>(tris[t].z)};

    computeSmoothNormals(mesh);
    proxy->meshes.push_back(std::move(mesh));
  }

  return proxy;
}

} // namespace sim::renderer
