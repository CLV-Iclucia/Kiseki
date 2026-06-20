// ============================================================================
// include/FluidSim/presets.h
// 预设工厂：一站式创建常见流体管线
// ============================================================================
#pragma once

#include <FluidSim/fluid-simulator.h>
#include <FluidSim/cpu/cpu-advector.h>
#include <FluidSim/cpu/cpu-projector.h>
#include <FluidSim/cpu/cpu-reconstructor.h>
#include <FluidSim/cpu/cpu-extrapolator.h>
#include <FluidSim/cpu/cpu-force.h>
#include <memory>
#include <optional>

namespace fluid {

enum class BackendType { CPU, GPU };

// ---- FreeSurface 配置 ----
struct FreeSurfaceConfig {
    FluidDomain domain;
    InitialFluid initialFluid;
    std::optional<Mesh> colliderMesh;

    cpu::AdvectorConfig advector;
    cpu::ProjectorConfig projector;
    cpu::ReconstructorConfig reconstructor;
    cpu::ForceConfig force;
    cpu::ExtrapolatorConfig extrapolator;

    Real maxCfl = 5.0;
};

// ---- Smoke 配置（预留）----
struct SmokeConfig {
    FluidDomain domain;

    cpu::ProjectorConfig projector;
    cpu::ForceConfig force;

    Real maxCfl = 5.0;
};

namespace presets {

/// 创建自由液面模拟管线 (CPU)
std::unique_ptr<FluidSimulator> createFreeSurface(
    const FreeSurfaceConfig& config,
    BackendType backend = BackendType::CPU);

/// 创建烟雾模拟管线 (预留)
// std::unique_ptr<FluidSimulator> createSmoke(
//     const SmokeConfig& config,
//     BackendType backend = BackendType::CPU);

} // namespace presets
} // namespace fluid
