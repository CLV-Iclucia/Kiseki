// ============================================================================
// build-particle-cell-ranges.hlsl
// Convert sorted particle cell keys into [start, end) ranges per grid cell.
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<uint> particleCellKeys : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> cellStart : register(u0);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> cellEnd : register(u1);

struct PushParams {
    uint numParticles;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.numParticles) {
        return;
    }

    uint key = particleCellKeys[i];
    if (i == 0 || particleCellKeys[i - 1] != key) {
        cellStart[key] = i;
    }
    if (i + 1 == pc.numParticles || particleCellKeys[i + 1] != key) {
        cellEnd[key] = i + 1;
    }
}
