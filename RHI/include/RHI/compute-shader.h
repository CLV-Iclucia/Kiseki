//
// compute-shader.h
// Typed compute shader objects: source metadata, compile options, Params type,
// and a cached compute pipeline behind one C++ type.
//

#pragma once

#include <RHI/device.h>
#include <RHI/reflection.h>
#include <RHI/shader-compile-options.h>
#include <RHI/shader-params.h>

#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sim::rhi {

struct ComputeShaderDefinition {
  std::filesystem::path source;
  std::string entryPoint = "main";
};

class ComputeShaderBase {
 public:
  ComputeShaderBase() = default;

  [[nodiscard]] bool valid() const noexcept { return m_pipeline.valid(); }
  explicit operator bool() const noexcept { return valid(); }

  [[nodiscard]] const std::filesystem::path& sourcePath() const noexcept {
    return m_source;
  }

  [[nodiscard]] const ReflectionInfo& reflection() const {
    if (!m_pipeline) {
      throw std::runtime_error(
          "cannot inspect an invalid typed compute shader");
    }
    return m_pipeline->reflection();
  }

 protected:
  void initialize(Device& device, ComputeShaderDefinition definition,
                  ShaderCompileOptions options);

  [[nodiscard]] const ReflectionInfo& pipelineReflection() const {
    return m_pipeline->reflection();
  }

 private:
  std::filesystem::path m_source;
  PipelineRef m_pipeline;

  friend class CommandList;
};

template <class Derived>
class ComputeShader : public ComputeShaderBase {
 public:
  ComputeShader() = default;

  explicit ComputeShader(Device& device) {
    ShaderCompileOptions options;

    if constexpr (requires { typename Derived::Permutation; }) {
      typename Derived::Permutation permutation{};
      Derived::modifyCompileOptions(options, permutation);
    } else if constexpr (requires {
                           Derived::modifyCompileOptions(options);
                         }) {
      Derived::modifyCompileOptions(options);
    }

    initializeTyped(device, std::move(options));
  }

  template <class D = Derived>
    requires requires(ShaderCompileOptions& options,
                      const typename D::Permutation& permutation) {
      D::modifyCompileOptions(options, permutation);
    }
  ComputeShader(Device& device,
                const typename D::Permutation& permutation) {
    ShaderCompileOptions options;
    D::modifyCompileOptions(options, permutation);
    initializeTyped(device, std::move(options));
  }

 private:
  void initializeTyped(Device& device, ShaderCompileOptions options) {
    auto definition = Derived::computeShaderDefinition();

    options.entryPoint = definition.entryPoint;
    options.stage = ShaderStage::Compute;
    options.targetBackend = device.backend();

    initialize(device, std::move(definition), std::move(options));

    if (valid()) {
      typename Derived::Params prototype;
      prototype._resolve(pipelineReflection());
    }
  }
};

}  // namespace sim::rhi

#define DECLARE_COMPUTE_SHADER(Type)                                          \
  using Super = ::sim::rhi::ComputeShader<Type>;                              \
  using Super::Super;                                                         \
  static ::sim::rhi::ComputeShaderDefinition computeShaderDefinition()

#define IMPLEMENT_COMPUTE_SHADER(Type, Source, Entry)                         \
  ::sim::rhi::ComputeShaderDefinition Type::computeShaderDefinition() {       \
    return {::std::filesystem::path(Source), Entry};                          \
  }
