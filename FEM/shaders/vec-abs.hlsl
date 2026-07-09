// vec-abs.hlsl — out[i] = |in[i]|  (full-precision double; HLSL abs() is float-only).
[[vk::binding(0, 0)]] RWStructuredBuffer<double> outv : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   inv  : register(t0);
struct PC { uint n; };
[[vk::push_constant]] PC pc;
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.n) return;
    double v = inv[i];
    outv[i] = v < 0.0 ? -v : v;
}
