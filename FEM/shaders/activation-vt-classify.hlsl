// ============================================================================
// activation-vt-classify.hlsl — module C, VT activation pass.
// One thread per VT candidate {v, t0, t1, t2}. Compute the point-triangle
// distance type + squared distance (shared logic, bit-identical to CPU), test
// against dHat^2, and on activation emit the unified constraint (kind+indices)
// into the combined candidate stream at e = streamOffset + i.
//
// Shared sort key = kind (0..3) for active, SH_CK_NONE (4) for inactive; a
// later stable sort by key reproduces CPU's PP|PE|PT|EE bucket layout.
// ============================================================================
#include <fem/shared/ipc-distance.h>
#include <fem/shared/ipc-activation.h>

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   keyOut     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   valOut     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>    idxOut     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>   countOut   : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<double>   x          : register(t0);
[[vk::binding(5, 0)]] StructuredBuffer<uint>     cand       : register(t1);
[[vk::binding(6, 0)]] StructuredBuffer<double>   dHatSqrBuf : register(t2);

struct PC { uint numVt; uint streamOffset; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.numVt) return;
    uint e = pc.streamOffset + i;
    valOut[e] = e;

    int v  = (int)cand[i * 4 + 0];
    int t0 = (int)cand[i * 4 + 1];
    int t1 = (int)cand[i * 4 + 2];
    int t2 = (int)cand[i * 4 + 3];

    double3 p = load_double3(x, v);
    double3 a = load_double3(x, t0);
    double3 b = load_double3(x, t1);
    double3 c = load_double3(x, t2);

    int    type = shDecidePointTriangleDistanceType(p, a, b, c);
    double d2   = shDistanceSqrPointTriangleByType(type, p, a, b, c);

    if (d2 < dHatSqrBuf[0]) {
        ShConstraintPair cp = shActivateVT(type, v, t0, t1, t2);
        keyOut[e]       = (uint)cp.kind;
        idxOut[e * 4 + 0] = cp.indices[0];
        idxOut[e * 4 + 1] = cp.indices[1];
        idxOut[e * 4 + 2] = cp.indices[2];
        idxOut[e * 4 + 3] = cp.indices[3];
        uint orig;
        InterlockedAdd(countOut[cp.kind], 1u, orig);
    } else {
        keyOut[e]       = SH_CK_NONE;
        idxOut[e * 4 + 0] = -1;
        idxOut[e * 4 + 1] = -1;
        idxOut[e * 4 + 2] = -1;
        idxOut[e * 4 + 3] = -1;
    }
}
