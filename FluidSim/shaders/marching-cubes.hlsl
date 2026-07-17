// ============================================================================
// marching-cubes.hlsl
// GPU iso-surface extraction from fluid SDF using the standard 256-case
// marching-cubes table mirrored from the CPU implementation.
// ============================================================================

#include "marching-cubes-tables.hlsli"

[[vk::binding(0, 0)]] Texture3D<float> fluidSdf : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> positions : register(u0);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> normals : register(u1);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> triangles : register(u2);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> counter : register(u3);

struct PushParams {
    uint gridSizeX;
    uint gridSizeY;
    uint gridSizeZ;
    float gridSpacing;
    float originX;
    float originY;
    float originZ;
    uint maxTriangles;
    uint maxVertices;
};
[[vk::push_constant]] PushParams pc;

// Must match FluidSim/src/cpu/rebuild-surface.cc vertexOffset.
static const uint3 kCornerOffset[8] = {
    uint3(0, 0, 0), uint3(1, 0, 0), uint3(1, 0, 1), uint3(0, 0, 1),
    uint3(0, 1, 0), uint3(1, 1, 0), uint3(1, 1, 1), uint3(0, 1, 1)
};

// Must match FluidSim/src/cpu/rebuild-surface.cc edgeConnection.
static const uint2 kEdgeConnection[12] = {
    uint2(0, 1), uint2(1, 2), uint2(3, 2), uint2(0, 3),
    uint2(4, 5), uint2(5, 6), uint2(7, 6), uint2(4, 7),
    uint2(0, 4), uint2(1, 5), uint2(2, 6), uint2(3, 7)
};

float3 origin()
{
    return float3(pc.originX, pc.originY, pc.originZ);
}

float sdfAt(uint3 c)
{
    c.x = min(c.x, pc.gridSizeX - 1);
    c.y = min(c.y, pc.gridSizeY - 1);
    c.z = min(c.z, pc.gridSizeZ - 1);
    return fluidSdf[c];
}

float3 worldAt(uint3 c)
{
    return origin() + (float3(c) + 0.5f) * pc.gridSpacing;
}

void storeFloat3(RWStructuredBuffer<float> dst, uint index, float3 value)
{
    uint base = index * 3;
    dst[base + 0] = value.x;
    dst[base + 1] = value.y;
    dst[base + 2] = value.z;
}

uint xEdgeId(uint x, uint y, uint z)
{
    return x + y * (pc.gridSizeX - 1) + z * (pc.gridSizeX - 1) * pc.gridSizeY;
}

uint yEdgeId(uint x, uint y, uint z)
{
    uint xEdgeCount = (pc.gridSizeX - 1) * pc.gridSizeY * pc.gridSizeZ;
    return xEdgeCount + x + y * pc.gridSizeX + z * pc.gridSizeX * (pc.gridSizeY - 1);
}

uint zEdgeId(uint x, uint y, uint z)
{
    uint xEdgeCount = (pc.gridSizeX - 1) * pc.gridSizeY * pc.gridSizeZ;
    uint yEdgeCount = pc.gridSizeX * (pc.gridSizeY - 1) * pc.gridSizeZ;
    return xEdgeCount + yEdgeCount + x + y * pc.gridSizeX + z * pc.gridSizeX * pc.gridSizeY;
}

uint globalEdgeVertexId(uint3 base, uint edge)
{
    switch (edge) {
    case 0: return xEdgeId(base.x,     base.y,     base.z);
    case 2: return xEdgeId(base.x,     base.y,     base.z + 1);
    case 4: return xEdgeId(base.x,     base.y + 1, base.z);
    case 6: return xEdgeId(base.x,     base.y + 1, base.z + 1);
    case 1: return zEdgeId(base.x + 1, base.y,     base.z);
    case 3: return zEdgeId(base.x,     base.y,     base.z);
    case 5: return zEdgeId(base.x + 1, base.y + 1, base.z);
    case 7: return zEdgeId(base.x,     base.y + 1, base.z);
    case 8: return yEdgeId(base.x,     base.y,     base.z);
    case 9: return yEdgeId(base.x + 1, base.y,     base.z);
    case 10: return yEdgeId(base.x + 1, base.y,    base.z + 1);
    case 11: return yEdgeId(base.x,    base.y,     base.z + 1);
    default: return 0;
    }
}

float3 sdfNormalAt(float3 p)
{
    float3 g = (p - origin()) / pc.gridSpacing - 0.5f;
    int3 c = int3(round(g));
    c.x = clamp(c.x, 0, int(pc.gridSizeX) - 1);
    c.y = clamp(c.y, 0, int(pc.gridSizeY) - 1);
    c.z = clamp(c.z, 0, int(pc.gridSizeZ) - 1);

    int x0 = max(c.x - 1, 0);
    int x1 = min(c.x + 1, int(pc.gridSizeX) - 1);
    int y0 = max(c.y - 1, 0);
    int y1 = min(c.y + 1, int(pc.gridSizeY) - 1);
    int z0 = max(c.z - 1, 0);
    int z1 = min(c.z + 1, int(pc.gridSizeZ) - 1);

    float gx = sdfAt(uint3(x1, c.y, c.z)) - sdfAt(uint3(x0, c.y, c.z));
    float gy = sdfAt(uint3(c.x, y1, c.z)) - sdfAt(uint3(c.x, y0, c.z));
    float gz = sdfAt(uint3(c.x, c.y, z1)) - sdfAt(uint3(c.x, c.y, z0));
    float3 n = float3(gx, gy, gz);
    float len = length(n);
    return (len > 1e-8f) ? n / len : float3(0.0f, 1.0f, 0.0f);
}

float3 interpolateVertex(float3 pa, float3 pb, float va, float vb)
{
    float denom = va - vb;
    float t = (abs(denom) > 1e-8f) ? saturate(va / denom) : 0.5f;
    return lerp(pa, pb, t);
}

void writeEdgeVertex(uint vertexId, float3 position)
{
    if (vertexId >= pc.maxVertices) {
        return;
    }
    storeFloat3(positions, vertexId, position);
    storeFloat3(normals, vertexId, sdfNormalAt(position));
}

void emitTriangle(uint a, uint b, uint c)
{
    uint tri;
    InterlockedAdd(counter[0], 1, tri);
    if (tri >= pc.maxTriangles) {
        return;
    }

    uint indexBase = tri * 3;
    triangles[indexBase + 0] = a;
    triangles[indexBase + 1] = b;
    triangles[indexBase + 2] = c;
}

void emitOrientedTriangle(uint a, uint b, uint c, float3 pa, float3 pb, float3 pc_)
{
    float3 na = sdfNormalAt(pa);
    float3 nb = sdfNormalAt(pb);
    float3 nc = sdfNormalAt(pc_);
    float3 faceNormal = normalize(cross(pb - pa, pc_ - pa));
    float3 avgNormal = normalize(na + nb + nc);
    if (dot(faceNormal, avgNormal) < 0.0f) {
        emitTriangle(a, c, b);
    } else {
        emitTriangle(a, b, c);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint cubeCountX = pc.gridSizeX - 1;
    uint cubeCountY = pc.gridSizeY - 1;
    uint cubeCountZ = pc.gridSizeZ - 1;
    uint cubeCount = cubeCountX * cubeCountY * cubeCountZ;
    uint idx = tid.x;
    if (idx >= cubeCount) {
        return;
    }

    uint3 base;
    base.x = idx % cubeCountX;
    base.y = (idx / cubeCountX) % cubeCountY;
    base.z = idx / (cubeCountX * cubeCountY);

    float3 cornerPos[8];
    float cornerSdf[8];
    uint cubeIndex = 0;

    [unroll]
    for (uint c = 0; c < 8; ++c) {
        uint3 corner = base + kCornerOffset[c];
        cornerPos[c] = worldAt(corner);
        cornerSdf[c] = sdfAt(corner);
        if (cornerSdf[c] < 0.0f) {
            cubeIndex |= (1u << c);
        }
    }

    if (kCubeEdgeFlags[cubeIndex] == 0) {
        return;
    }

    uint edgeVertexId[12];
    float3 edgeVertexPos[12];
    [unroll]
    for (uint e = 0; e < 12; ++e) {
        if ((kCubeEdgeFlags[cubeIndex] & (1 << e)) != 0) {
            uint2 endpoints = kEdgeConnection[e];
            uint a = endpoints.x;
            uint b = endpoints.y;
            edgeVertexId[e] = globalEdgeVertexId(base, e);
            float3 edgePosition = interpolateVertex(
                cornerPos[a], cornerPos[b], cornerSdf[a], cornerSdf[b]);
            edgeVertexPos[e] = edgePosition;
            writeEdgeVertex(edgeVertexId[e], edgePosition);
        }
    }

    [unroll]
    for (uint tri = 0; tri < 5; ++tri) {
        int e0 = kTriangleConnectionTable3D[cubeIndex][tri * 3 + 0];
        if (e0 < 0) {
            break;
        }
        uint ue0 = uint(e0);
        uint e1 = uint(kTriangleConnectionTable3D[cubeIndex][tri * 3 + 1]);
        uint e2 = uint(kTriangleConnectionTable3D[cubeIndex][tri * 3 + 2]);
        emitOrientedTriangle(
            edgeVertexId[ue0], edgeVertexId[e1], edgeVertexId[e2],
            edgeVertexPos[ue0], edgeVertexPos[e1], edgeVertexPos[e2]);
    }
}
