#include "interactive-renderer.h"

#include "render-backend.h"
#include "render-backend-factory.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace ksk::renderer {

namespace {

constexpr const char* IMGUI_GLSL_VERSION = "#version 330";

bool isKeyDown(GLFWwindow* window, int key) {
  return glfwGetKey(window, key) == GLFW_PRESS;
}

} // namespace

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

  initializeUi();
  setupInputCallbacks();
  m_lastInputTime = glfwGetTime();

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
  renderUi(scene);
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

  double currentTime = glfwGetTime();
  float deltaTime = static_cast<float>(currentTime - m_lastInputTime);
  m_lastInputTime = currentTime;
  m_deltaTime = deltaTime;

  handleKeyboardMovement(deltaTime);
  updateCameraFromInput();
  return !m_backend || m_backend->present();
}

void InteractiveRenderer::cleanup() {
  if (m_backend) {
    m_backend->cleanup();
    m_backend.reset();
  }

  cleanupUi();

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
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (action != GLFW_PRESS) return;

    switch (key) {
      case GLFW_KEY_F1:
        renderer->m_showUi = !renderer->m_showUi;
        break;

      case GLFW_KEY_R:
        if (renderer->uiWantsKeyboard()) break;
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
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (renderer->uiWantsMouse()) return;

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
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (renderer->uiWantsMouse()) {
      renderer->m_mousePressed = false;
      renderer->m_lastMouseX = xpos;
      renderer->m_lastMouseY = ypos;
      return;
    }

    if (renderer->m_mousePressed) {
      double dx = xpos - renderer->m_lastMouseX;
      double dy = ypos - renderer->m_lastMouseY;

      renderer->m_yaw -= static_cast<float>(dx) * renderer->m_mouseSensitivity;
      renderer->m_pitch -= static_cast<float>(dy) * renderer->m_mouseSensitivity;
      renderer->m_pitch = std::max(-89.0f, std::min(89.0f, renderer->m_pitch));

      renderer->m_lastMouseX = xpos;
      renderer->m_lastMouseY = ypos;
    }
  });

  glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset,
                                     double yoffset) {
    auto* renderer = static_cast<InteractiveRenderer*>(glfwGetWindowUserPointer(window));
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (renderer->uiWantsMouse()) return;

    float zoomFactor = 1.0f - static_cast<float>(yoffset) * renderer->m_zoomSensitivity;
    renderer->m_distance *= zoomFactor;
    renderer->m_distance = std::max(0.01f, renderer->m_distance);
  });

  glfwSetCharCallback(m_window, [](GLFWwindow* window, unsigned int codepoint) {
    ImGui_ImplGlfw_CharCallback(window, codepoint);
  });
}

void InteractiveRenderer::initializeUi() {
  if (!m_window || m_uiInitialized) return;

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;

  ImGui_ImplGlfw_InitForOpenGL(m_window, false);
  ImGui_ImplOpenGL3_Init(IMGUI_GLSL_VERSION);
  m_uiInitialized = true;
}

void InteractiveRenderer::renderUi(const SceneProxy& scene) {
  if (!m_uiInitialized) return;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (m_showUi) {
    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Renderer")) {
      ImGui::Text("Frame %d  Time %.3f", scene.frameIndex, scene.simulationTime);
      ImGui::Text("FPS %.1f  dt %.3f ms",
                  m_deltaTime > 0.0f ? 1.0f / m_deltaTime : 0.0f,
                  m_deltaTime * 1000.0f);
      ImGui::Separator();

      ImGui::Text("Camera");
      ImGui::DragFloat3("Target", &m_camera.target.x, 0.01f);
      ImGui::DragFloat("Distance", &m_distance, 0.05f, 0.01f, 10000.0f);
      ImGui::SliderFloat("Yaw", &m_yaw, -180.0f, 180.0f);
      ImGui::SliderFloat("Pitch", &m_pitch, -89.0f, 89.0f);
      ImGui::SliderFloat("FOV", &m_camera.fov, 10.0f, 100.0f);

      if (ImGui::Button("Reset Camera")) {
        m_cameraInitialized = false;
      }

      ImGui::Separator();
      ImGui::Text("Controls");
      ImGui::DragFloat("Move Speed", &m_keyboardMoveSpeed, 0.05f, 0.01f, 100.0f);
      ImGui::DragFloat("Fast Multiplier", &m_keyboardFastMultiplier, 0.05f,
                       1.0f, 20.0f);
      ImGui::DragFloat("Mouse Sensitivity", &m_mouseSensitivity, 0.01f,
                       0.01f, 5.0f);
      ImGui::DragFloat("Zoom Sensitivity", &m_zoomSensitivity, 0.01f,
                       0.01f, 1.0f);

      ImGui::Separator();
      ImGui::Text("Scene");
      ImGui::Text("Meshes: %zu", scene.meshes.size());
      ImGui::Text("Wireframes: %zu", scene.wireframes.size());
      ImGui::Text("Particles: %zu", scene.particles.size());
      ImGui::Text("F1 toggles this panel");
    }
    ImGui::End();
  }

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void InteractiveRenderer::cleanupUi() {
  if (!m_uiInitialized) return;

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  m_uiInitialized = false;
}

bool InteractiveRenderer::uiWantsKeyboard() const {
  return m_uiInitialized && m_showUi && ImGui::GetIO().WantCaptureKeyboard;
}

bool InteractiveRenderer::uiWantsMouse() const {
  return m_uiInitialized && m_showUi && ImGui::GetIO().WantCaptureMouse;
}

void InteractiveRenderer::handleKeyboardMovement(float deltaTime) {
  if (!m_window || deltaTime <= 0.0f || uiWantsKeyboard()) return;

  glm::vec3 forward = glm::normalize(m_camera.target - m_camera.position);
  glm::vec3 right = glm::normalize(glm::cross(forward, m_camera.up));
  glm::vec3 movement(0.0f);

  if (isKeyDown(m_window, GLFW_KEY_W)) movement += forward;
  if (isKeyDown(m_window, GLFW_KEY_S)) movement -= forward;
  if (isKeyDown(m_window, GLFW_KEY_D)) movement += right;
  if (isKeyDown(m_window, GLFW_KEY_A)) movement -= right;
  if (isKeyDown(m_window, GLFW_KEY_E)) movement += m_camera.up;
  if (isKeyDown(m_window, GLFW_KEY_Q)) movement -= m_camera.up;

  float movementLength = glm::length(movement);
  if (movementLength <= 0.0f) return;

  float speed = m_keyboardMoveSpeed * std::max(m_distance, 0.1f);
  if (isKeyDown(m_window, GLFW_KEY_LEFT_SHIFT) ||
      isKeyDown(m_window, GLFW_KEY_RIGHT_SHIFT)) {
    speed *= m_keyboardFastMultiplier;
  }

  m_camera.target += (movement / movementLength) * speed * deltaTime;
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
