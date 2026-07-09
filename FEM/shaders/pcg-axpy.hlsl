// ============================================================================
// pcg-axpy.hlsl — y += (sign * alpha) * x   (double).  alpha read from device.
// One thread per vertex (3 scalar components). Vectors are double[N*3].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> y        : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   x        : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   alphaBuf : register(t1);

struct PC { uint n; float sign; };  // n = number of vertices
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.n) return;
    double a = alphaBuf[0] * (double)pc.sign;
    double3 result = load_double3(y, tid.x);
    result += load_double3(x, tid.x) * a;
    store_double3(y, tid.x, result);
}
