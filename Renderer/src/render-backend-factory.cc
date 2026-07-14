#include "render-backend-factory.h"

#include "gl-render-backend.h"

#include <Renderer/renderer.h>

namespace ksk::renderer {

std::unique_ptr<RenderBackend> createRenderBackend(RenderBackendKind kind) {
  switch (kind) {
    case RenderBackendKind::Default:
    case RenderBackendKind::OpenGL:
      return std::make_unique<GLRenderBackend>();

    case RenderBackendKind::RHI:
      // TODO: return RhiRenderBackend once the Renderer layer links RHI.
      return std::make_unique<GLRenderBackend>();
  }

  return std::make_unique<GLRenderBackend>();
}

} // namespace ksk::renderer
