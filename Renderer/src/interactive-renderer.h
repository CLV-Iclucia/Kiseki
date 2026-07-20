#pragma once

#include <Renderer/renderer.h>
#include <glm/glm.hpp>

#include <memory>

struct GLFWwindow;

namespace ksk::renderer {

class RenderBackend;

struct SceneBounds {
  glm::vec3 center{0.0f};
  float radius = 1.0f;
};

class InteractiveRenderer final : public Renderer {
protected:
  void initialize(const RendererConfig& config) override;
  void drawFrame(const SceneProxy& scene) override;
  void cleanup() override;
  bool pollAndSwap() override;

private:
  ::GLFWwindow* m_window = nullptr;
  std::unique_ptr<RenderBackend> m_backend;

  bool m_mousePressed = false;
  double m_lastMouseX = 0;
  double m_lastMouseY = 0;
  float m_yaw = -60.0f;
  float m_pitch = 25.0f;
  float m_distance = 5.0f;
  double m_lastInputTime = 0.0;
  float m_deltaTime = 0.0f;
  float m_keyboardMoveSpeed = 2.5f;
  float m_keyboardFastMultiplier = 4.0f;
  float m_mouseSensitivity = 0.5f;
  float m_zoomSensitivity = 0.15f;
  bool m_showUi = true;
  bool m_uiInitialized = false;
  bool m_cameraInitialized = false;

  glm::mat4 computeViewMatrix() const;
  glm::mat4 computeProjectionMatrix() const;

  void setupInputCallbacks();
  void initializeUi();
  void renderUi(const SceneProxy& scene);
  void cleanupUi();
  bool uiWantsKeyboard() const;
  bool uiWantsMouse() const;
  void handleKeyboardMovement(float deltaTime);
  void updateCameraFromInput();

  SceneBounds computeSceneBounds(const SceneProxy& scene) const;
};

} // namespace ksk::renderer
