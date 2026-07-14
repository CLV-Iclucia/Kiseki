#pragma once

#include "render-backend.h"

#include <string>
#include <unordered_map>

struct GLFWwindow;

namespace ksk::renderer {

class GLRenderBackend final : public RenderBackend {
public:
  void configureWindow() override;
  bool initialize(const RenderBackendInit& init) override;
  void renderFrame(const SceneProxy& scene,
                   const RenderFrameContext& context) override;
  bool present() override;
  void cleanup() override;

private:
  struct GLMeshState {
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;
    unsigned int nbo = 0;
    int indexCount = 0;
    size_t vertexCount = 0;
  };

  ::GLFWwindow* m_window = nullptr;
  std::unordered_map<std::string, GLMeshState> m_meshCache;

  unsigned int m_meshShader = 0;
  unsigned int m_wireShader = 0;
  unsigned int m_particleShader = 0;

  unsigned int m_groundGridVao = 0;
  unsigned int m_groundGridVbo = 0;
  int m_groundGridVertexCount = 0;

  void uploadMesh(const MeshProxy& mesh);
  void drawMesh(const MeshProxy& mesh);
  void drawWireframe(const WireframeProxy& wf);
  void drawParticles(const ParticleProxy& particles,
                     const RenderFrameContext& context);
  void drawGroundGrid(const RenderFrameContext& context);
  void buildGroundGrid();
};

} // namespace ksk::renderer
