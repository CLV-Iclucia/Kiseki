// vec-axpy.hlsl — y[i] += a * x[i]  (flat double arrays, host-supplied scalar a).
[[vk::binding(0, 0)]] RWStructuredBuffer<double> y : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   x : register(t0);
struct PC { double a; uint n; };
[[vk::push_constant]] PC pc;
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;
    y[i] += pc.a * x[i];
}
