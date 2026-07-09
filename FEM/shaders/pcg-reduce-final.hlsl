// ============================================================================
// pcg-reduce-final.hlsl — single-workgroup final sum of partials (double).
// ============================================================================
[[vk::binding(0, 0)]] StructuredBuffer<double>   partial : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<double> result  : register(u0);

struct PC { uint numGroups; };
[[vk::push_constant]] PC pc;

groupshared double sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    double v = 0.0;
    for (uint i = gid; i < pc.numGroups; i += 256) v += partial[i];
    sdata[gid] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) sdata[gid] += sdata[gid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gid == 0) result[0] = sdata[0];
}
