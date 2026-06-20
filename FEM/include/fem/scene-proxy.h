#pragma once
#include <Renderer/scene-proxy.h>
#include <fem/system.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace sim::renderer {

/// 从 FEM System 构建渲染帧（公共工具函数）。
/// 遍历所有 Primitive，提取表面三角形 + 当前顶点位置。
/// 消除了每个 App 都重复实现 buildSceneProxy 的问题。
inline std::unique_ptr<SceneProxy> buildSceneProxyFromSystem(
    const fem::System& system, int frameIndex) {
  auto proxy = std::make_unique<SceneProxy>();
  proxy->frameIndex = frameIndex;
  proxy->simulationTime = static_cast<float>(system.currentTime());

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

  return proxy;
}

} // namespace sim::renderer
