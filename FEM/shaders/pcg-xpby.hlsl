// ============================================================================
// pcg-xpby.hlsl — p = z + beta * p   (double).  beta read from device.
// One thread per vertex (3 scalar components). Vectors are double[N*3].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> p       : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   z       : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   betaBuf : register(t1);

struct PC { uint n; };  // n = number of vertices
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.n) return;
    double b = betaBuf[0];
    double3 result = load_double3(z, tid.x)
                   + load_double3(p, tid.x) * b;
    store_double3(p, tid.x, result);
}
