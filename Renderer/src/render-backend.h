#pragma once

#include <Renderer/scene-proxy.h>
#include <glm/glm.hpp>

namespace ksk::renderer {

struct RenderBackendInit {
  void* nativeWindow = nullptr;
  int width = 1;
  int height = 1;
  bool vsync = true;
  bool headless = false;
};

struct RenderFrameContext {
  glm::mat4 view{1.0f};
  glm::mat4 projection{1.0f};
  glm::vec3 cameraPosition{0.0f};
  glm::vec3 cameraTarget{0.0f};
  float cameraDistance = 1.0f;
  int framebufferWidth = 1;
  int framebufferHeight = 1;
};

class RenderBackend {
public:
  virtual ~RenderBackend() = default;

  virtual void configureWindow() = 0;
  virtual bool initialize(const RenderBackendInit& init) = 0;
  virtual void renderFrame(const SceneProxy& scene,
                           const RenderFrameContext& context) = 0;
  virtual bool present() = 0;
  virtual void cleanup() = 0;
};

} // namespace ksk::renderer
