// ============================================================================
// body-force.hlsl — Apply gravity to V-face velocities (float32)
// ============================================================================

[[vk::binding(0, 0)]] RWStructuredBuffer<float> vGrid : register(u0);

struct PushParams {
    float dt;
    float gravityY;
    uint  numVFaces;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numVFaces) return;
    vGrid[idx] += pc.gravityY * pc.dt;
}
