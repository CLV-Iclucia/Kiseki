// ============================================================================
// include/FluidSim/gpu/gpu-marching-cubes.h
// GPU surface extraction from the reconstructed fluid SDF.
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <FluidSim/gpu/gpu-backend.h>
#include <FluidSim/gpu/gpu-shaders.h>
#include <RHI/rhi.h>

namespace fluid::gpu {

class GPUMarchingCubes {
public:
    GPUMarchingCubes(ksk::rhi::Device& device, const GPUGridState& grid);

    void execute(ksk::rhi::CommandList& cmd, const GPUGridState& grid);
    bool readback(ksk::rhi::Device& device, FluidSurfaceMesh& out);

private:
    static constexpr uint32_t kMaxTrianglesPerCell = 5;

    MarchingCubesCS marchingCubes_;
    ksk::rhi::BufferRef positions_;
    ksk::rhi::BufferRef normals_;
    ksk::rhi::BufferRef triangles_;
    ksk::rhi::BufferRef counter_;
    ksk::rhi::BufferRef readbackPositions_;
    ksk::rhi::BufferRef readbackNormals_;
    ksk::rhi::BufferRef readbackTriangles_;
    ksk::rhi::BufferRef readbackCounter_;
    uint32_t maxTriangles_{0};
    uint32_t maxVertices_{0};
};

} // namespace fluid::gpu
