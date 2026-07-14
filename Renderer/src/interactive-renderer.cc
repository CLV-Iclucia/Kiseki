#include "interactive-renderer.h"

#include "render-backend.h"
#include "render-backend-factory.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace ksk::renderer {

void InteractiveRenderer::initialize(const RendererConfig& config) {
  m_config = config;

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return;
  }

  m_backend = createRenderBackend(config.backend);
  m_backend->configureWindow();

  if (config.headless) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  }

  m_window = glfwCreateWindow(config.windowWidth, config.windowHeight,
                              config.windowTitle.c_str(), nullptr, nullptr);
  if (!m_window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return;
  }

  if (!m_backend->initialize({
          .nativeWindow = m_window,
          .width = config.windowWidth,
          .height = config.windowHeight,
          .vsync = config.vsync,
          .headless = config.headless,
      })) {
    std::cerr << "Failed to initialize render backend" << std::endl;
    glfwDestroyWindow(m_window);
    m_window = nullptr;
    glfwTerminate();
    return;
  }

  setupInputCallbacks();

  std::cout << "InteractiveRenderer initialized successfully" << std::endl;
}

void InteractiveRenderer::drawFrame(const SceneProxy& scene) {
  if (!m_cameraInitialized) {
    auto bounds = computeSceneBounds(scene);
    m_camera.target = bounds.center;

    float halfFovTan = std::tan(glm::radians(m_camera.fov * 0.5f));
    m_distance = (bounds.radius / halfFovTan) * 1.6f;
    m_distance = std::max(m_distance, bounds.radius * 4.0f);

    m_pitch = 25.0f;
    m_yaw = -60.0f;
    m_cameraInitialized = true;

    updateCameraFromInput();
  }

  int framebufferWidth = 0;
  int framebufferHeight = 0;
  glfwGetFramebufferSize(m_window, &framebufferWidth, &framebufferHeight);
  if (framebufferWidth <= 0 || framebufferHeight <= 0 || !m_backend) {
    return;
  }

  RenderFrameContext context{
      .view = computeViewMatrix(),
      .projection = computeProjectionMatrix(),
      .cameraPosition = m_camera.position,
      .cameraTarget = m_camera.target,
      .cameraDistance = m_distance,
      .framebufferWidth = framebufferWidth,
      .framebufferHeight = framebufferHeight,
  };

  m_backend->renderFrame(scene, context);
}

glm::mat4 InteractiveRenderer::computeViewMatrix() const {
  return glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
}

glm::mat4 InteractiveRenderer::computeProjectionMatrix() const {
  int width = 1;
  int height = 1;
  glfwGetFramebufferSize(m_window, &width, &height);
  width = std::max(width, 1);
  height = std::max(height, 1);
  float aspect = static_cast<float>(width) / static_cast<float>(height);
  return glm::perspective(glm::radians(m_camera.fov), aspect,
                          m_camera.nearPlane, m_camera.farPlane);
}

bool InteractiveRenderer::pollAndSwap() {
  if (!m_window) return false;

  glfwPollEvents();

  if (glfwWindowShouldClose(m_window)) {
    return false;
  }

  updateCameraFromInput();
  return !m_backend || m_backend->present();
}

void InteractiveRenderer::cleanup() {
  if (m_backend) {
    m_backend->cleanup();
    m_backend.reset();
  }

  if (m_window) {
    glfwDestroyWindow(m_window);
    m_window = nullptr;
  }

  glfwTerminate();
  std::cout << "InteractiveRenderer cleaned up" << std::endl;
}

void InteractiveRenderer::setupInputCallbacks() {
  if (!m_window) return;

  glfwSetWindowUserPointer(m_window, this);

  glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode,
                                  int action, int mods) {
    auto* renderer = static_cast<InteractiveRenderer*>(glfwGetWindowUserPointer(window));
    if (action != GLFW_PRESS) return;

    switch (key) {
      case GLFW_KEY_R:
        renderer->m_cameraInitialized = false;
        break;

      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;

      default:
        break;
    }
  });

  glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button,
                                          int action, int mods) {
    auto* renderer = static_cast<InteractiveRenderer*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      renderer->m_mousePressed = (action == GLFW_PRESS);
      if (renderer->m_mousePressed) {
        glfwGetCursorPos(window, &renderer->m_lastMouseX,
                         &renderer->m_lastMouseY);
      }
    }
  });

  glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos,
                                        double ypos) {
    auto* renderer = static_cast<InteractiveRenderer*>(glfwGetWindowUserPointer(window));
    if (renderer->m_mousePressed) {
      double dx = xpos - renderer->m_lastMouseX;
      double dy = ypos - renderer->m_lastMouseY;

      renderer->m_yaw -= static_cast<float>(dx) * 0.5f;
      renderer->m_pitch -= static_cast<float>(dy) * 0.5f;
      renderer->m_pitch = std::max(-89.0f, std::min(89.0f, renderer->m_pitch));

      renderer->m_lastMouseX = xpos;
      renderer->m_lastMouseY = ypos;
    }
  });

  glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset,
                                     double yoffset) {
    auto* renderer = static_cast<InteractiveRenderer*>(glfwGetWindowUserPointer(window));
    float zoomFactor = 1.0f - static_cast<float>(yoffset) * 0.15f;
    renderer->m_distance *= zoomFactor;
    renderer->m_distance = std::max(0.01f, renderer->m_distance);
  });
}

void InteractiveRenderer::updateCameraFromInput() {
  float yawRad = glm::radians(m_yaw);
  float pitchRad = glm::radians(m_pitch);

  m_camera.position.x =
      m_camera.target.x + m_distance * std::cos(pitchRad) * std::cos(yawRad);
  m_camera.position.y = m_camera.target.y + m_distance * std::sin(pitchRad);
  m_camera.position.z =
      m_camera.target.z + m_distance * std::cos(pitchRad) * std::sin(yawRad);

  if (m_inputCallback) {
    m_inputCallback(m_camera);
  }
}

SceneBounds InteractiveRenderer::computeSceneBounds(const SceneProxy& scene) const {
  SceneBounds bounds;

  if (scene.meshes.empty() && scene.particles.empty()) return bounds;

  glm::vec3 minP(std::numeric_limits<float>::max());
  glm::vec3 maxP(std::numeric_limits<float>::lowest());

  for (const auto& mesh : scene.meshes) {
    for (const auto& p : mesh.positions) {
      glm::vec3 pos(p.x, p.y, p.z);
      minP = glm::min(minP, pos);
      maxP = glm::max(maxP, pos);
    }
  }

  for (const auto& ps : scene.particles) {
    for (const auto& p : ps.positions) {
      glm::vec3 pos(p.x, p.y, p.z);
      minP = glm::min(minP, pos);
      maxP = glm::max(maxP, pos);
    }
  }

  if (minP.x > maxP.x) return bounds;

  bounds.center = (minP + maxP) * 0.5f;
  bounds.radius = glm::length(maxP - bounds.center);
  bounds.radius = std::max(bounds.radius, 0.1f);

  return bounds;
}

std::unique_ptr<Renderer> createRenderer(const RendererConfig& config) {
  auto renderer = std::make_unique<InteractiveRenderer>();
  renderer->setConfig(config);
  return renderer;
}

} // namespace ksk::renderer
