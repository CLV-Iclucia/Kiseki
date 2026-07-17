// ============================================================================
// src/gpu/gpu-marching-cubes.cc
// Conservative GPU surface extraction from the reconstructed fluid SDF.
// ============================================================================

#include <FluidSim/gpu/gpu-marching-cubes.h>
#include <Core/profiler.h>
#include <RHI/rhi.h>

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>

namespace fluid::gpu {

using namespace ksk::rhi;

GPUMarchingCubes::GPUMarchingCubes(Device& device, const GPUGridState& grid)
    : marchingCubes_(device)
{
    const uint32_t nx = static_cast<uint32_t>(std::max(grid.gridSize.x, 0));
    const uint32_t ny = static_cast<uint32_t>(std::max(grid.gridSize.y, 0));
    const uint32_t nz = static_cast<uint32_t>(std::max(grid.gridSize.z, 0));
    const uint64_t cellCount = static_cast<uint64_t>(std::max(grid.gridSize.x - 1, 0)) *
                               static_cast<uint64_t>(std::max(grid.gridSize.y - 1, 0)) *
                               static_cast<uint64_t>(std::max(grid.gridSize.z - 1, 0));
    const uint64_t xEdgeCount = static_cast<uint64_t>(std::max(grid.gridSize.x - 1, 0)) *
                                ny * nz;
    const uint64_t yEdgeCount = static_cast<uint64_t>(nx) *
                                std::max(grid.gridSize.y - 1, 0) * nz;
    const uint64_t zEdgeCount = static_cast<uint64_t>(nx) * ny *
                                std::max(grid.gridSize.z - 1, 0);
    maxTriangles_ = static_cast<uint32_t>(
        std::min<uint64_t>(cellCount * kMaxTrianglesPerCell,
                           std::numeric_limits<uint32_t>::max() / 3));
    maxVertices_ = static_cast<uint32_t>(
        std::min<uint64_t>(xEdgeCount + yEdgeCount + zEdgeCount,
                           std::numeric_limits<uint32_t>::max()));

    const size_t vectorBytes = static_cast<size_t>(maxVertices_) * sizeof(Vec3f);
    const size_t indexBytes = static_cast<size_t>(maxTriangles_) * sizeof(Vec3u);

    positions_ = createDeviceLocalBuffer(device, vectorBytes, "fluid-surface-positions");
    normals_ = createDeviceLocalBuffer(device, vectorBytes, "fluid-surface-normals");
    triangles_ = createDeviceLocalBuffer(device, indexBytes, "fluid-surface-triangles");
    counter_ = createDeviceLocalBuffer(device, sizeof(uint32_t), "fluid-surface-counter");

    readbackPositions_ = device.createBuffer({
        .sizeBytes = std::max<size_t>(vectorBytes, 4),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-surface-readback-positions",
    });
    readbackNormals_ = device.createBuffer({
        .sizeBytes = std::max<size_t>(vectorBytes, 4),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-surface-readback-normals",
    });
    readbackTriangles_ = device.createBuffer({
        .sizeBytes = std::max<size_t>(indexBytes, 4),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-surface-readback-triangles",
    });
    readbackCounter_ = device.createBuffer({
        .sizeBytes = sizeof(uint32_t),
        .visibility = BufferDesc::Visibility::Readback,
        .usage = BufferDesc::TransferDst,
        .debugName = "fluid-surface-readback-counter",
    });
}

void GPUMarchingCubes::execute(CommandList& cmd, const GPUGridState& grid) {
    SIM_PROFILE_FUNCTION();

    if (!marchingCubes_.valid() || !grid.fluidSdfImg || maxTriangles_ == 0) {
        return;
    }

    cmd.fillBuffer(counter_, 0);
    cmd.memoryBarrier(BarrierDesc::StageTransfer, BarrierDesc::StageComputeShader,
                      BarrierDesc::AccessTransferWrite,
                      BarrierDesc::AccessShaderRead | BarrierDesc::AccessShaderWrite);

    const uint32_t nx = static_cast<uint32_t>(std::max(grid.gridSize.x - 1, 0));
    const uint32_t ny = static_cast<uint32_t>(std::max(grid.gridSize.y - 1, 0));
    const uint32_t nz = static_cast<uint32_t>(std::max(grid.gridSize.z - 1, 0));
    const uint32_t cellCount = nx * ny * nz;
    if (cellCount == 0) {
        return;
    }
    const uint32_t cellGroups = (cellCount + 255) / 256;

    MarchingCubesCS::Params params;
    params.fluidSdf = ImageBinding{grid.fluidSdfImg, grid.sdfSampler};
    params.positions = positions_;
    params.normals = normals_;
    params.triangles = triangles_;
    params.counter = counter_;
    params.gridSizeX = static_cast<uint32_t>(grid.gridSize.x);
    params.gridSizeY = static_cast<uint32_t>(grid.gridSize.y);
    params.gridSizeZ = static_cast<uint32_t>(grid.gridSize.z);
    params.gridSpacing = grid.gridSpacing;
    params.originX = grid.originX;
    params.originY = grid.originY;
    params.originZ = grid.originZ;
    params.maxTriangles = maxTriangles_;
    params.maxVertices = maxVertices_;
    cmd.dispatch(marchingCubes_, params, cellGroups, 1, 1);
}

bool GPUMarchingCubes::readback(Device& device, FluidSurfaceMesh& out) {
    SIM_PROFILE_FUNCTION();

    out = {};
    if (maxTriangles_ == 0) {
        return false;
    }

    const size_t vectorBytes = static_cast<size_t>(maxVertices_) * sizeof(Vec3f);
    const size_t indexBytes = static_cast<size_t>(maxTriangles_) * sizeof(Vec3u);
    auto cmd = device.beginCommands(QueueType::Transfer);
    std::array<BufferCopy, 1> counterCopy{{{0, 0, sizeof(uint32_t)}}};
    std::array<BufferCopy, 1> vectorCopy{{{0, 0, vectorBytes}}};
    std::array<BufferCopy, 1> indexCopy{{{0, 0, indexBytes}}};
    cmd->copyBuffer(counter_, readbackCounter_, counterCopy);
    cmd->copyBuffer(positions_, readbackPositions_, vectorCopy);
    cmd->copyBuffer(normals_, readbackNormals_, vectorCopy);
    cmd->copyBuffer(triangles_, readbackTriangles_, indexCopy);
    cmd->end();
    device.submitAndWait(*cmd, QueueType::Transfer);

    auto counterData = readbackCounter_->mapTyped<uint32_t>();
    const uint32_t triangleCount = std::min(counterData.empty() ? 0u : counterData[0],
                                            maxTriangles_);
    readbackCounter_->unmap();
    if (triangleCount == 0) {
        return false;
    }

    auto positions = readbackPositions_->mapTyped<Vec3f>();
    auto normals = readbackNormals_->mapTyped<Vec3f>();
    auto triangles = readbackTriangles_->mapTyped<Vec3u>();

    std::unordered_map<uint32_t, uint32_t> vertexMap;
    vertexMap.reserve(static_cast<size_t>(triangleCount) * 3);
    out.positions.reserve(static_cast<size_t>(triangleCount) * 3);
    out.normals.reserve(static_cast<size_t>(triangleCount) * 3);
    out.triangles.reserve(triangleCount);

    auto compactVertex = [&](uint32_t sourceVertex) {
        if (auto it = vertexMap.find(sourceVertex); it != vertexMap.end()) {
            return it->second;
        }
        const uint32_t compactIndex = static_cast<uint32_t>(out.positions.size());
        vertexMap.emplace(sourceVertex, compactIndex);
        out.positions.push_back(positions[sourceVertex]);
        out.normals.push_back(normals[sourceVertex]);
        return compactIndex;
    };

    for (uint32_t tri = 0; tri < triangleCount; ++tri) {
        const Vec3u srcTri = triangles[tri];
        if (srcTri.x >= maxVertices_ || srcTri.y >= maxVertices_ ||
            srcTri.z >= maxVertices_) {
            continue;
        }
        const Vec3u compactTri{
            compactVertex(srcTri.x),
            compactVertex(srcTri.y),
            compactVertex(srcTri.z),
        };

        if (compactTri.x != compactTri.y &&
            compactTri.y != compactTri.z &&
            compactTri.z != compactTri.x) {
            out.triangles.push_back(compactTri);
        }
    }

    for (Vec3f& n : out.normals) {
        const float len = glm::length(n);
        n = (len > 1e-8f) ? n / len : Vec3f(0.0f, 1.0f, 0.0f);
    }

    readbackPositions_->unmap();
    readbackNormals_->unmap();
    readbackTriangles_->unmap();
    return !out.triangles.empty();
}

} // namespace fluid::gpu
