// ============================================================================
// pcg-scalar-div.hlsl — result[0] = num[0] / denom[0]  (double, on-GPU).
// ============================================================================
[[vk::binding(0, 0)]] StructuredBuffer<double>   num    : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   denom  : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<double> result : register(u0);

[numthreads(1, 1, 1)]
void main() {
    double d = denom[0];
    result[0] = (abs(d) > 1e-300) ? (num[0] / d) : 0.0;
}
