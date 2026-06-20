// ============================================================================
// cg-saxpy.hlsl — CG: SAXPY (y[i] += alpha * x[i]) (float32)
// alpha is stored in reduceBuf[0] (result of previous dot product reduction).
// ============================================================================

[[vk::binding(0, 0)]] RWStructuredBuffer<float> y        : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   x        : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<float>   alphaBuf : register(t1);

struct PushParams {
    uint  numCells;
    float sign;   // +1 for p += alpha*s, -1 for r -= alpha*z
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numCells) return;
    float alpha = alphaBuf[0];
    y[idx] += pc.sign * alpha * x[idx];
}
