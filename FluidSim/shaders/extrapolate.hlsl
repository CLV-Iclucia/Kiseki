// ============================================================================
// extrapolate.hlsl — Extrapolate velocities into invalid face cells (float32)
// One Jacobi step: invalid ← average of valid 6-connected neighbors.
// Must be called for each of u/v/w faces separately via the same shader
// (dispatch with faceCount for that component).
// ============================================================================

[[vk::binding(0, 0)]] StructuredBuffer<float>   gridIn    : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> gridOut   : register(u0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>    validIn   : register(t1);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>  validOut  : register(u1);

struct PushParams {
    uint faceResX;
    uint faceResY;
    uint faceResZ;
    uint numFaces;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numFaces) return;

    uint3 fr = uint3(pc.faceResX, pc.faceResY, pc.faceResZ);

    if (validIn[idx] != 0u) {
        // Already valid: copy through unchanged
        gridOut[idx]  = gridIn[idx];
        validOut[idx] = 1u;
        return;
    }

    // Decode 3D index
    uint3 f;
    f.x = idx % fr.x;
    f.y = (idx / fr.x) % fr.y;
    f.z = idx / (fr.x * fr.y);

    float sum   = 0.0f;
    int   count = 0;

    // 6-connected neighbors
    uint3 nb;
    #define TRY_NB(nx2, ny2, nz2) \
        nb = uint3(nx2, ny2, nz2); \
        if (all(nb >= uint3(0,0,0)) && all(nb < fr)) { \
            uint ni = nb.x + nb.y * fr.x + nb.z * fr.x * fr.y; \
            if (validIn[ni] != 0u) { sum += gridIn[ni]; count++; } \
        }

    if (f.x > 0)       { TRY_NB(f.x-1, f.y, f.z) }
    if (f.x+1 < fr.x)  { TRY_NB(f.x+1, f.y, f.z) }
    if (f.y > 0)       { TRY_NB(f.x, f.y-1, f.z) }
    if (f.y+1 < fr.y)  { TRY_NB(f.x, f.y+1, f.z) }
    if (f.z > 0)       { TRY_NB(f.x, f.y, f.z-1) }
    if (f.z+1 < fr.z)  { TRY_NB(f.x, f.y, f.z+1) }

    if (count > 0) {
        gridOut[idx]  = sum / float(count);
        validOut[idx] = 1u;
    } else {
        gridOut[idx]  = 0.0f;
        validOut[idx] = 0u;
    }
}
