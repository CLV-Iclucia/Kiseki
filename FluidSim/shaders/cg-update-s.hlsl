// ============================================================================
// cg-update-s.hlsl — CG: s = z + beta * s  (beta = sigmaNew / sigmaOld)
// betaBuf[0] = sigmaNew, sigmaOldBuf[0] = sigmaOld.
// ============================================================================

[[vk::binding(0, 0)]] RWStructuredBuffer<float> s          : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   z          : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<float>   sigmaNewBuf: register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<float>   sigmaOldBuf: register(t2);

struct PushParams {
    uint numCells;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numCells) return;

    float sigmaOld = sigmaOldBuf[0];
    float beta = (abs(sigmaOld) > 1e-30f) ? (sigmaNewBuf[0] / sigmaOld) : 0.0f;
    s[idx] = z[idx] + beta * s[idx];
}
