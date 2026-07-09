//
// shader-utils.h
// Free-function utilities that bridge ShaderCompiler and Device.
//
// These helpers are intentionally NOT members of Device or ShaderCompiler:
//   • Device doesn't know about ShaderCompiler (keeps device.h lean).
//   • ShaderCompiler doesn't know about Device or Pipeline.
//   • Callers that only need raw bytecode compilation never pay for the
//     Device include; callers that need pipelines include this file.
//
// Usage:
//   #include <RHI/shader-utils.h>      // or just #include <RHI/rhi.h>
//
//   auto pso = sim::rhi::compileComputePipeline(device, compiler,
//                  "kernels/advect.hlsl");
//   if (!pso.valid())  /* handle error */;
//

#pragma once

#include <RHI/device.h>
#include <RHI/pipeline.h>
#include <RHI/shader-compiler.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace sim::rhi {

// ---------------------------------------------------------------------------
// compileComputePipeline
//
// One-call helper: compile an HLSL file → create shader → create pipeline.
//
//   device   — owns the resulting shader and pipeline objects
//   compiler — must be pre-bound to a backend (e.g. via
//              device.createShaderCompiler()); targetBackend may be left unset
//   path     — absolute or relative HLSL source file; parent directory is
//              automatically added as an #include search path
//   entry    — shader entry point (default "main")
//
// Returns an invalid PipelineRef on any error; the compiler logs the
// diagnostic via spdlog at error level.
// ---------------------------------------------------------------------------
inline PipelineRef compileComputePipeline(
    Device&                      device,
    ShaderCompiler&              compiler,
    const std::filesystem::path& path,
    ShaderCompileOptions          options)
{
    // A pipeline belongs to `device`, so selecting its bytecode backend
    // independently is always an error. Standalone compilation can still
    // select a backend through ShaderCompiler::compileHlsl* directly.
    options.stage = ShaderStage::Compute;
    options.targetBackend = device.backend();

    auto compiled = compiler.compileHlslFile(path, options);
    if (!compiled) return {};

    auto shader = device.createShader(
        compiled->bytecode, ShaderStage::Compute, options.entryPoint);
    if (!shader.valid()) return {};

    return device.createComputePipeline({.shader = shader});
}

inline PipelineRef compileComputePipeline(
    Device&                      device,
    ShaderCompiler&              compiler,
    const std::filesystem::path& path,
    std::string_view             entry = "main")
{
    ShaderCompileOptions options;
    options.entryPoint = std::string(entry);
    return compileComputePipeline(device, compiler, path, options);
}

} // namespace sim::rhi
