// ============================================================================
// dirichlet.hlsl — Zero out normal velocity on domain boundary faces (float32)
// ============================================================================
#include "fluid-common.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> uGrid : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> vGrid : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> wGrid : register(u2);

struct PushParams {
    uint3 gridSize;
    uint  _pad;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint3 gs = pc.gridSize;

    // U faces on X=0 and X=nx boundaries
    uint nXBound = gs.y * gs.z;
    if (idx < nXBound * 2) {
        bool isXmax = (idx >= nXBound);
        uint jk = isXmax ? (idx - nXBound) : idx;
        uint y  = jk % gs.y;
        uint z  = jk / gs.y;
        uint x  = isXmax ? gs.x : 0;
        uGrid[idxUFace(uint3(x, y, z), gs)] = 0.0f;
        return;
    }
    idx -= nXBound * 2;

    // V faces on Y=0 and Y=ny boundaries
    uint nYBound = gs.x * gs.z;
    if (idx < nYBound * 2) {
        bool isYmax = (idx >= nYBound);
        uint ik = isYmax ? (idx - nYBound) : idx;
        uint x  = ik % gs.x;
        uint z  = ik / gs.x;
        uint y  = isYmax ? gs.y : 0;
        vGrid[idxVFace(uint3(x, y, z), gs)] = 0.0f;
        return;
    }
    idx -= nYBound * 2;

    // W faces on Z=0 and Z=nz boundaries
    uint nZBound = gs.x * gs.y;
    if (idx < nZBound * 2) {
        bool isZmax = (idx >= nZBound);
        uint ij = isZmax ? (idx - nZBound) : idx;
        uint x  = ij % gs.x;
        uint y  = ij / gs.x;
        uint z  = isZmax ? gs.z : 0;
        wGrid[idxWFace(uint3(x, y, z), gs)] = 0.0f;
    }
}
