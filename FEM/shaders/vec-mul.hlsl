// vec-mul.hlsl — out[i] = a[i] * b[i]  (elementwise; for dot via Sum reduction).
[[vk::binding(0, 0)]] RWStructuredBuffer<double> outv : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   a    : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<double>   b    : register(t1);
struct PC { uint n; };
[[vk::push_constant]] PC pc;
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;
    outv[i] = a[i] * b[i];
}
