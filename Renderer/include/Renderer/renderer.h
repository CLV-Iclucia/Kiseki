#pragma once
#include <Renderer/scene-proxy.h>
#include <Renderer/frame-queue.h>
#include <memory>
#include <thread>
#include <functional>
#include <atomic>

namespace ksk::renderer {

struct RendererConfig {
  int windowWidth = 1280;
  int windowHeight = 720;
  std::string windowTitle = "Kiseki";
  bool vsync = true;
  bool headless = false;
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
      // 1. 非阻塞尝试获取新帧；如果有就更新，没有就复用上一帧
      if (auto newFrame = m_queue.tryPop())
        currentFrame = std::move(newFrame);

      // 如果还没有任何帧（模拟尚未推送第一帧），阻塞等一次
      if (!currentFrame) {
        currentFrame = m_queue.pop();
        if (!currentFrame) break; // shutdown
      }

      // 2. 处理用户输入
      if (m_inputCallback)
        m_inputCallback(m_camera);

      // 3. 渲染当前帧
      currentFrame->camera = m_camera;
      drawFrame(*currentFrame);

      // 4. swap + poll events（保持窗口响应）
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

/// 工厂函数
std::unique_ptr<Renderer> createRenderer(const RendererConfig& config = {});

} // namespace ksk::renderer
