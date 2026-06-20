// ============================================================================
// src/presets.cc
// 预设工厂实现
// ============================================================================

#include <FluidSim/presets.h>
#include <FluidSim/cpu/cpu-context.h>
#include <FluidSim/cpu/sdf.h>
#include <Core/error.h>
#include <iostream>

namespace fluid::presets {

std::unique_ptr<FluidSimulator> createFreeSurface(
    const FreeSurfaceConfig& config, BackendType backend)
{
    if (backend == BackendType::CPU) {
        // 1. 创建 Context（构造时分配所有网格/SDF 等成员）
        auto ctx = std::make_unique<cpu::CPUFluidContext>(config.domain);

        // 2. 填充初始数据（直接写成员）
        ctx->positions = config.initialFluid.positions;
        ctx->velocities = config.initialFluid.velocities;

        // 3. 上传 collider SDF（如果有）
        if (config.colliderMesh.has_value()) {
            std::cout << "[presets] Building collider SDF..." << std::endl;
            spatify::Array3D<int> closest(ctx->colliderSdf->width(),
                                          ctx->colliderSdf->height(),
                                          ctx->colliderSdf->depth());
            spatify::Array3D<int> intersection_cnt(ctx->colliderSdf->width(),
                                                    ctx->colliderSdf->height(),
                                                    ctx->colliderSdf->depth());
            manifold2SDF(3, closest, intersection_cnt,
                         *config.colliderMesh, ctx->colliderSdf.get());
            std::cout << "[presets] Done." << std::endl;
        }

        // 4. 创建 Solver（构造时传入 ctx 引用 + 各自 config）
        auto advImpl = std::make_shared<cpu::CPUAdvector>(*ctx, config.advector);
        auto advP2G  = std::make_shared<cpu::CPUAdvectorP2G>(advImpl);
        auto extrap  = std::make_shared<cpu::CPUExtrapolator>(*ctx, config.extrapolator);
        auto recon   = std::make_shared<cpu::CPUReconstructor>(*ctx, config.reconstructor);
        auto force   = std::make_shared<cpu::CPUForceApplicator>(*ctx, config.force);
        auto proj    = std::make_shared<cpu::CPUProjector>(*ctx, config.projector);
        auto advG2P  = std::make_shared<cpu::CPUAdvectorG2P>(advImpl);

        // 5. 一次构造 Simulator（管线从此不可变）
        // 执行顺序:
        //   P2G scatter → Extrapolate → Reconstruct → Force → Project → G2P
        return std::make_unique<FluidSimulator>(
            std::move(ctx),
            std::vector<std::shared_ptr<FluidModularSolver>>{
                advP2G, extrap, recon, force, proj, advG2P
            }
        );
    }
    else {
        // GPU 版本需要 Device 和 ShaderCompiler 参数，
        // 由单独的 createFreeSurfaceGPU 函数处理
        SIM_THROW("GPU backend via presets::createFreeSurface requires "
                  "createFreeSurfaceGPU(device, compiler, config)");
    }
}

} // namespace fluid::presets
