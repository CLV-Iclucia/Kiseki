// ============================================================================
// lbvh-morton.hlsl — 10-bit-per-axis Morton code from each primitive's AABB
// centre, computed in double to bit-match spatify::LBVH::mortonCode (Real=double).
//   keys[i] = encodeMorton10bit(quantize(centre_i))   (30-bit in a uint)
//   vals[i] = i                                        (original primitive index)
// Quantization mirrors CPU exactly: v = max((c-lo)/(hi-lo)*1024, 0); floor; &0x3FF.
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   keys       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   vals       : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<double>   aabbLo     : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<double>   aabbHi     : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<double>   sceneBound : register(t2); // [6]

struct PC { uint n; };
[[vk::push_constant]] PC pc;

uint part1by2(uint x) {            // == spatify::morton10BitEncode
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
}
uint encodeMorton10(uint x, uint y, uint z) {
    return (part1by2(x) << 2) | (part1by2(y) << 1) | part1by2(z);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;

    double3 center =
        (load_double3(aabbLo, i) + load_double3(aabbHi, i)) * 0.5;
    double3 sceneLo = load_double3(sceneBound, 0u);
    double3 sceneHi = load_double3(sceneBound, 1u);

    double vx = max((center.x - sceneLo.x) / (sceneHi.x - sceneLo.x) * 1024.0, 0.0);
    double vy = max((center.y - sceneLo.y) / (sceneHi.y - sceneLo.y) * 1024.0, 0.0);
    double vz = max((center.z - sceneLo.z) / (sceneHi.z - sceneLo.z) * 1024.0, 0.0);

    uint xi = (uint)(int)vx;       // truncate toward zero, like CPU (int) cast
    uint yi = (uint)(int)vy;
    uint zi = (uint)(int)vz;

    keys[i] = encodeMorton10(xi, yi, zi);
    vals[i] = i;
}
