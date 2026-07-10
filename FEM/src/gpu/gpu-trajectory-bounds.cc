// ============================================================================
// FEM/src/gpu/gpu-trajectory-bounds.cc
// ============================================================================
#include <fem/gpu/gpu-trajectory-bounds.h>

#include <spdlog/spdlog.h>

namespace ksk::fem::gpu {

using namespace ksk::rhi;

#ifndef FEM_SHADER_DIR
#define FEM_SHADER_DIR "."
#endif

static constexpr uint32_t kWG = 256;
static uint32_t groups(uint32_t n) { return (n + kWG - 1) / kWG; }

IMPLEMENT_COMPUTE_SHADER(
    TrajBoundCS, FEM_SHADER_DIR "/traj-bound.hlsl", "main");

GpuTrajectoryBounds::GpuTrajectoryBounds(Device& device)
    : device_(device), shader_(device)
{
    if (!shader_)
        spdlog::error("[GpuTrajectoryBounds] pipeline compilation failed");
}

void GpuTrajectoryBounds::record(CommandList& cmd, const BufferRef& x, const BufferRef& p,
                                 const BufferRef& conn, const BufferRef& alphaBuf,
                                 const BufferRef& outLo, const BufferRef& outHi,
                                 uint32_t numPrims, uint32_t vertsPerPrim) {
    if (!shader_ || numPrims == 0) return;
    TrajBoundCS::Params params;
    params.outLo = outLo; params.outHi = outHi;
    params.x = x; params.p = p; params.conn = conn; params.alphaBuf = alphaBuf;
    params.numPrims = numPrims; params.vertsPerPrim = vertsPerPrim;
    cmd.dispatch(shader_, params, groups(numPrims), 1, 1);
}

void GpuTrajectoryBounds::compute(const BufferRef& x, const BufferRef& p,
                                  const BufferRef& conn, const BufferRef& alphaBuf,
                                  const BufferRef& outLo, const BufferRef& outHi,
                                  uint32_t numPrims, uint32_t vertsPerPrim) {
    if (!shader_ || numPrims == 0) return;
    auto cmd = device_.beginCommands(QueueType::Compute);
    record(*cmd, x, p, conn, alphaBuf, outLo, outHi, numPrims, vertsPerPrim);
    device_.submitAndWait(*cmd, QueueType::Compute);
}

} // namespace ksk::fem::gpu
