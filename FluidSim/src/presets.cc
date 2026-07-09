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
        auto ctx = std::make_unique<cpu::CPUFluidContext>(config.domain);

        ctx->positions = config.initialFluid.positions;
        ctx->velocities = config.initialFluid.velocities;

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

        auto advImpl = std::make_shared<cpu::CPUAdvector>(*ctx, config.advector);
        auto advP2G  = std::make_shared<cpu::CPUAdvectorP2G>(advImpl);
        auto extrap  = std::make_shared<cpu::CPUExtrapolator>(*ctx, config.extrapolator);
        auto recon   = std::make_shared<cpu::CPUReconstructor>(*ctx, config.reconstructor);
        auto force   = std::make_shared<cpu::CPUForceApplicator>(*ctx, config.force);
        auto proj    = std::make_shared<cpu::CPUProjector>(*ctx, config.projector);
        auto advG2P  = std::make_shared<cpu::CPUAdvectorG2P>(advImpl);

        //   P2G scatter → Extrapolate → Reconstruct → Force → Project → G2P
        return std::make_unique<FluidSimulator>(
            std::move(ctx),
            std::vector<std::shared_ptr<FluidModularSolver>>{
                advP2G, extrap, recon, force, proj, advG2P
            }
        );
    }
    else {
        SIM_THROW("GPU backend via presets::createFreeSurface requires "
                  "a GPUFluidContext created with an RHI device");
    }
}

} // namespace fluid::presets
