// ============================================================================
// smooth-sdf.hlsl — Jacobi smooth + extrapolate SDF
// ============================================================================

[[vk::binding(0, 0)]] Texture3D<float> srcSdf : register(t0);
[[vk::binding(1, 0)]] RWTexture3D<float> dstSdf : register(u0);

struct PushParams {
    uint3 gridSize;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint nc = pc.gridSize.x * pc.gridSize.y * pc.gridSize.z;
    if (idx >= nc) return;

    uint3 c;
    c.x = idx % pc.gridSize.x;
    c.y = (idx / pc.gridSize.x) % pc.gridSize.y;
    c.z = idx / (pc.gridSize.x * pc.gridSize.y);

    float sum = srcSdf[c];
    int count = 1;

    if (c.x > 0)                { sum += srcSdf[uint3(c.x-1, c.y, c.z)]; count++; }
    if (c.x+1 < pc.gridSize.x)  { sum += srcSdf[uint3(c.x+1, c.y, c.z)]; count++; }
    if (c.y > 0)                { sum += srcSdf[uint3(c.x, c.y-1, c.z)]; count++; }
    if (c.y+1 < pc.gridSize.y)  { sum += srcSdf[uint3(c.x, c.y+1, c.z)]; count++; }
    if (c.z > 0)                { sum += srcSdf[uint3(c.x, c.y, c.z-1)]; count++; }
    if (c.z+1 < pc.gridSize.z)  { sum += srcSdf[uint3(c.x, c.y, c.z+1)]; count++; }

    float smoothed = sum / float(count);
    dstSdf[c] = (srcSdf[c] < smoothed) ? srcSdf[c] : smoothed;
}
