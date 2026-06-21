// ============================================================================
// cg-scalar-div.hlsl — Compute scalar quotient: out[0] = num[0] / denom[0]
// Used to compute CG alpha = sigma / dot(s,z) on GPU without CPU readback.
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<float>   num   : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   denom : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> result : register(u0);

[numthreads(1, 1, 1)]
void main() {
    float d = denom[0];
    result[0] = (abs(d) > 1e-30f) ? (num[0] / d) : 0.0f;
}
