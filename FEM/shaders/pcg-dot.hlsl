// ============================================================================
// pcg-dot.hlsl — block vector dot product (double) → per-workgroup partials.
// One thread per vertex (3 scalar components). Vectors are double[N*3].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] StructuredBuffer<double>   a       : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   b       : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<double> partial : register(u0);

struct PC { uint n; };  // n = number of vertices
[[vk::push_constant]] PC pc;

groupshared double sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex, uint3 grp : SV_GroupID) {
    double v = 0.0;
    if (tid.x < pc.n) {
        double3 av = load_double3(a, tid.x);
        double3 bv = load_double3(b, tid.x);
        v = av.x * bv.x + av.y * bv.y + av.z * bv.z;
    }
    sdata[gid] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) sdata[gid] += sdata[gid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gid == 0) partial[grp.x] = sdata[0];
}
