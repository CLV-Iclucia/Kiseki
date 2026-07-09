//
// shader-compile-options.h
// Lightweight shader compilation configuration shared by the public typed
// shader API and the low-level ShaderCompiler interface.
//

#pragma once

#include <RHI/backend.h>
#include <RHI/shader.h>

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sim::rhi {

struct ShaderCompileOptions {
  std::string entryPoint = "main";
  ShaderStage stage = ShaderStage::Compute;

  // Optional per-call override. Typed shaders always replace this with their
  // owning device's backend.
  std::optional<Backend> targetBackend;

  // Preprocessor defines: each pair is (name, value). Empty value emits
  // `-D NAME`.
  std::vector<std::pair<std::string, std::string>> defines;

  // Additional include search paths (`-I`).
  std::vector<std::filesystem::path> includeDirs;

  // Populate CompiledShader::disassembly when supported by the backend.
  bool generateDisassembly = false;

  // -Od -Zi when true; -O3 otherwise.
  bool enableDebugInfo = false;
};

}  // namespace sim::rhi
