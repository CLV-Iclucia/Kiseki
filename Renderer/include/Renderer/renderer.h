#pragma once
#include <Renderer/scene-proxy.h>
#include <Renderer/frame-queue.h>
#include <memory>
#include <thread>
#include <functional>
#include <atomic>

namespace ksk::renderer {

enum class RenderBackendKind {
  Default,
  OpenGL,
  RHI,
};

struct RendererConfig {
  int windowWidth = 1280;
  int windowHeight = 720;
  std::string windowTitle = "Kiseki";
  bool vsync = true;
  bool headless = false;
  RenderBackendKind backend = RenderBackendKind::Default;
};

class Renderer {
public:
  virtual ~Renderer() = default;

  [[nodiscard]] FrameQueue& queue() { return m_queue; }

  void setConfig(const RendererConfig& config) { m_config = config; }

  void runOnCurrentThread() {
    m_running = true;
    renderLoop();
  }

  void shutdown() {
    m_running = false;
    m_queue.shutdown();
  }

  [[nodiscard]] bool isRunning() const { return m_running; }

  void setInputCallback(std::function<void(CameraState&)> cb) { m_inputCallback = std::move(cb); }

protected:
  virtual void initialize(const RendererConfig& config) = 0;
  virtual void drawFrame(const SceneProxy& scene) = 0;
  virtual void cleanup() = 0;
  virtual bool pollAndSwap() = 0;

  RendererConfig m_config;
  CameraState m_camera;
  std::function<void(CameraState&)> m_inputCallback;

private:
  void renderLoop() {
    initialize(m_config);

    std::unique_ptr<SceneProxy> currentFrame;

    while (m_running) {
      if (auto newFrame = m_queue.tryPop())
        currentFrame = std::move(newFrame);

      if (!currentFrame) {
        currentFrame = m_queue.pop();
        if (!currentFrame) break; // shutdown
      }

      if (m_inputCallback)
        m_inputCallback(m_camera);

      currentFrame->camera = m_camera;
      drawFrame(*currentFrame);

      if (!pollAndSwap()) {
        m_running = false;
        m_queue.shutdown();
      }
    }

    cleanup();
  }

  FrameQueue m_queue;
  std::atomic<bool> m_running{false};
};

std::unique_ptr<Renderer> createRenderer(const RendererConfig& config = {});

} // namespace ksk::renderer
