// ============================================================================
// FEM/include/fem/gpu/gpu-broad-phase.h
// GPU broad phase: query each vertex against the triangle LBVH (VT) and each
// edge against the edge LBVH (EE), producing candidate collision pairs via the
// classic count -> scan -> write pattern (deterministic, no atomics).
//
// Mirrors CPU IpcIntegrator::compute{VertexTriangle,EdgeEdge}CollisionPairs:
//   * query AABB = primitive trajectory AABB dilated by dHat,
//   * BVH leaf AABB = raw trajectory AABB,
//   * VT skips triangles containing the vertex; EE skips adjacent edges,
//   * EE is NOT i<j deduped (each edge queries all edges) — matches CPU.
//
// Candidate payloads (global vertex block indices):
//   VT: {vertex, triV0, triV1, triV2}   EE: {edgeA0, edgeA1, edgeB0, edgeB1}
// ============================================================================
#pragma once

#include <RHI/rhi.h>
#include <RPK/scan.h>
#include <fem/gpu/gpu-lbvh.h>

#include <filesystem>
#include <memory>

namespace ksk::fem::gpu {

SHADER_PARAMS_BEGIN(BroadCountParams)
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, counts);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, nodeLo);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, nodeHi);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, lch);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, rch);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, sortedIdx);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, qLo);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, qHi);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, conn);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, dHatBuf);
    SHADER_PARAM_SCALAR(uint32_t,            numQueries);
    SHADER_PARAM_SCALAR(uint32_t,            numPrims);
SHADER_PARAMS_END();

SHADER_PARAMS_BEGIN(BroadWriteParams)
    SHADER_PARAM_UAV   (ksk::rhi::BufferRef, out);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, nodeLo);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, nodeHi);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, lch);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, rch);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, sortedIdx);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, qLo);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, qHi);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, conn);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, dHatBuf);
    SHADER_PARAM_SRV   (ksk::rhi::BufferRef, offsets);
    SHADER_PARAM_SCALAR(uint32_t,            numQueries);
    SHADER_PARAM_SCALAR(uint32_t,            numPrims);
SHADER_PARAMS_END();

// ============================================================================
// GpuBroadPhase
// ============================================================================
class GpuBroadPhase {
public:
    GpuBroadPhase(ksk::rhi::Device& device,
                  ksk::rhi::ShaderCompiler& compiler,
                  const std::filesystem::path& shaderDir = {});

    [[nodiscard]] bool valid() const { return valid_; }

    // VT: query each of `numVerts` vertices against the triangle BVH.
    //   qVertLo/Hi : double[numVerts*3]   vertex trajectory AABBs (raw)
    //   triConn    : uint[numTris*3]      global vertex indices
    //   dHatBuf    : double[1]
    // Returns candidate count; pairs available in vtPairs() as uint[count*4].
    uint32_t queryVT(const GPULBVH& triBvh,
                     const ksk::rhi::BufferRef& qVertLo,
                     const ksk::rhi::BufferRef& qVertHi,
                     const ksk::rhi::BufferRef& triConn,
                     const ksk::rhi::BufferRef& dHatBuf,
                     uint32_t numVerts);

    // EE: query each of `numEdges` edges against the edge BVH.
    //   qEdgeLo/Hi : double[numEdges*3]   edge trajectory AABBs (raw; same as the
    //                buffers used to build edgeBvh)
    //   edgeConn   : uint[numEdges*2]
    uint32_t queryEE(const GPULBVH& edgeBvh,
                     const ksk::rhi::BufferRef& qEdgeLo,
                     const ksk::rhi::BufferRef& qEdgeHi,
                     const ksk::rhi::BufferRef& edgeConn,
                     const ksk::rhi::BufferRef& dHatBuf,
                     uint32_t numEdges);

    [[nodiscard]] const ksk::rhi::BufferRef& vtPairs() const { return vtOut_; }
    [[nodiscard]] const ksk::rhi::BufferRef& eePairs() const { return eeOut_; }

private:
    ksk::rhi::Device& device_;
    bool valid_ = false;

    std::unique_ptr<ksk::rpk::Scan> scan_;
    ksk::rhi::PipelineRef psoVtCount_, psoVtWrite_, psoEeCount_, psoEeWrite_;

    ksk::rhi::BufferRef counts_, offsets_, vtOut_, eeOut_;
    uint32_t capQuery_ = 0, capVt_ = 0, capEe_ = 0;

    // Shared query driver: returns candidate count, leaves pairs in `outBuf`.
    uint32_t runQuery(const ksk::rhi::PipelineRef& psoCount,
                      const ksk::rhi::PipelineRef& psoWrite,
                      const GPULBVH& bvh,
                      const ksk::rhi::BufferRef& qLo, const ksk::rhi::BufferRef& qHi,
                      const ksk::rhi::BufferRef& conn, const ksk::rhi::BufferRef& dHatBuf,
                      uint32_t numQueries, ksk::rhi::BufferRef& outBuf, uint32_t& outCap);

    void ensureQueryCap(uint32_t numQueries);
    uint32_t readbackUintAt(const ksk::rhi::BufferRef& src, uint32_t index);
};

} // namespace ksk::fem::gpu
