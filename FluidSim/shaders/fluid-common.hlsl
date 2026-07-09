// ============================================================================
// fluid-common.hlsl — GPU FluidSim shared types & utilities (float32)
// ============================================================================

#include <RHI/structured-buffer-access.hlsli>

// ---- Particle SOA helpers ----
// Particles are stored as SOA (Structure of Arrays):
//   positions:  StructuredBuffer<float> (tightly-packed, 3 floats per particle)
//   velocities: StructuredBuffer<float> (tightly-packed, 3 floats per particle)
// Access: float3 pos = float3(positions[idx*3], positions[idx*3+1], positions[idx*3+2])
// NOTE: Do NOT use StructuredBuffer<float3> — its stride is 16 bytes, not 12!

// ---- Staggered grid indexing ----
uint idxCell(uint3 c, uint3 gs) {
    return c.x + c.y * gs.x + c.z * gs.y * gs.x;
}
uint idxUFace(uint3 f, uint3 gs) {
    return f.x + f.y * (gs.x + 1) + f.z * gs.y * (gs.x + 1);
}
uint idxVFace(uint3 f, uint3 gs) {
    return f.x + f.y * gs.x + f.z * (gs.y + 1) * gs.x;
}
uint idxWFace(uint3 f, uint3 gs) {
    return f.x + f.y * gs.x + f.z * gs.y * gs.x;
}

// ---- Grid coordinate helpers ----
// World position → continuous grid coordinates (cell-centred)
float3 worldToGrid(float3 world, float3 origin, float gsp) {
    return (world - origin) / gsp - 0.5f;
}

// Floor to integer cell index
int3 gridToCellI(float3 gridPos) {
    return int3(floor(gridPos));
}

// Fractional part within cell
float3 cellFrac(float3 gridPos) {
    return gridPos - floor(gridPos);
}

float3 worldToUVW(float3 world, float3 origin, float3 gs, float gsp) {
    return (world - origin) / (gs * gsp);
}

float3 cellCenter(uint3 c, float3 origin, float gsp) {
    return origin + (float3(c) + 0.5f) * gsp;
}

uint3 posToCell(float3 world, float3 origin, float gsp) {
    return uint3(floor((world - origin) / gsp));
}

// ---- Trilinear hat kernel (for P2G, support = 2 cells) ----
float hatKernel(float r) {
    float ar = abs(r);
    return max(1.0f - ar, 0.0f);
}

// ---- Atomic float add via uint CAS (HLSL float atomics workaround) ----
void atomicAddFloat(RWStructuredBuffer<uint> buf, uint index, float value) {
    uint expected = buf[index];
    [allow_uav_condition]
    while (true) {
        uint desired = asuint(asfloat(expected) + value);
        uint original;
        InterlockedCompareExchange(buf[index], expected, desired, original);
        if (original == expected) break;
        expected = original;
    }
}
