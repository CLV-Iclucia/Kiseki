#pragma once

#include <memory>

namespace ksk::renderer {

class RenderBackend;
enum class RenderBackendKind;

std::unique_ptr<RenderBackend> createRenderBackend(RenderBackendKind kind);

} // namespace ksk::renderer
