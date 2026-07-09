// ============================================================================
// pcg-jacobi-apply-dot.hlsl — fused preconditioner apply + dot(r,z) (double).
//   z[i] = invDiag[i] * r[i];  partial = sum r[i]·z[i]
// Vectors are scalar double[N*3]; invDiag is column-major double[N*9].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> z       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<double> partial : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<double>   r       : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<double>   invDiag : register(t1);

struct PC { uint n; };  // n = number of vertices
[[vk::push_constant]] PC pc;

groupshared double sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex, uint3 grp : SV_GroupID) {
    double contrib = 0.0;
    if (tid.x < pc.n) {
        uint i = tid.x;
        double3 rv = load_double3(r, i);
        double3x3 inverse = load_double3x3_col_major(invDiag, i);
        double3 zv;
        zv.x = inverse[0][0] * rv.x + inverse[0][1] * rv.y
             + inverse[0][2] * rv.z;
        zv.y = inverse[1][0] * rv.x + inverse[1][1] * rv.y
             + inverse[1][2] * rv.z;
        zv.z = inverse[2][0] * rv.x + inverse[2][1] * rv.y
             + inverse[2][2] * rv.z;
        store_double3(z, i, zv);
        contrib = rv.x * zv.x + rv.y * zv.y + rv.z * zv.z;
    }
    sdata[gid] = contrib;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (gid < s) sdata[gid] += sdata[gid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gid == 0) partial[grp.x] = sdata[0];
}
