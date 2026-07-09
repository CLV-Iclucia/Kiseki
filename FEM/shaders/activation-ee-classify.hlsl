// ============================================================================
// activation-ee-classify.hlsl — module C, EE activation pass.
// One thread per EE candidate {a0, a1, b0, b1}. Same as the VT pass but with
// edge-edge distance type / squared distance / activation remap. Writes into
// the combined candidate stream at e = streamOffset + i (streamOffset = numVt
// so all VT entries precede all EE entries, matching CPU's scatter order).
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

struct PC { uint numEe; uint streamOffset; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.numEe) return;
    uint e = pc.streamOffset + i;
    valOut[e] = e;

    int a0 = (int)cand[i * 4 + 0];
    int a1 = (int)cand[i * 4 + 1];
    int b0 = (int)cand[i * 4 + 2];
    int b1 = (int)cand[i * 4 + 3];

    double3 ea0 = load_double3(x, a0);
    double3 ea1 = load_double3(x, a1);
    double3 eb0 = load_double3(x, b0);
    double3 eb1 = load_double3(x, b1);

    int    type = shDecideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1);
    double d2   = shDistanceSqrEdgeEdgeByType(type, ea0, ea1, eb0, eb1);

    if (d2 < dHatSqrBuf[0]) {
        ShConstraintPair cp = shActivateEE(type, a0, a1, b0, b1);
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
