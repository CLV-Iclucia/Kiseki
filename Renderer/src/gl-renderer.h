#pragma once
#include <Renderer/renderer.h>
#include <unordered_map>
#include <string>
#include <glm/glm.hpp>

struct GLFWwindow;

namespace sim::renderer {

struct SceneBounds {
  glm::vec3 center{0.0f};
  float radius = 1.0f;
};

class GLRenderer final : public Renderer {
protected:
  void initialize(const RendererConfig& config) override;
  void drawFrame(const SceneProxy& scene) override;
  void cleanup() override;
  bool pollAndSwap() override;

private:
  struct GLMeshState {
    unsigned int vao = 0, vbo = 0, ebo = 0, nbo = 0;
    int indexCount = 0;
    size_t vertexCount = 0; // 用于检测是否需要重新分配 buffer
  };

  // 按 name 缓存 GPU 资源，SceneProxy 更新时同步
  std::unordered_map<std::string, GLMeshState> m_meshCache;

  // GLFW window handle
  ::GLFWwindow* m_window = nullptr;

  // Shader programs
  unsigned int m_meshShader = 0;
  unsigned int m_wireShader = 0;
  unsigned int m_particleShader = 0;

  // 相机交互状态
  bool m_mousePressed = false;
  double m_lastMouseX = 0, m_lastMouseY = 0;
  float m_yaw = -60.0f;     // 水平角度（F2.3: 原 -90.0f）
  float m_pitch = 25.0f;    // 垂直角度（F2.3: 原 0.0f）
  float m_distance = 5.0f;   // 距离目标的距离

  // 首帧初始化标记
  bool m_cameraInitialized = false;

  // 地面参考网格（F3.2）
  unsigned int m_groundGridVao = 0;
  unsigned int m_groundGridVbo = 0;
  int m_groundGridVertexCount = 0;

  void uploadMesh(const MeshProxy& mesh);
  void drawMesh(const MeshProxy& mesh);
  void drawWireframe(const WireframeProxy& wf);
  void drawParticles(const ParticleProxy& particles);
  glm::mat4 computeViewMatrix() const;
  glm::mat4 computeProjectionMatrix() const;

  // 相机交互
  void setupInputCallbacks();
  void updateCameraFromInput();

  // 场景包围盒计算（仅首帧初始化用）
  SceneBounds computeSceneBounds(const SceneProxy& scene) const;

  // 地面参考网格（F3.2）
  void drawGroundGrid(const glm::mat4& view, const glm::mat4& projection);
  void buildGroundGrid();
};

} // namespace sim::renderer
