// ============================================================================
// pcg-jacobi-setup.hlsl — Block-Jacobi preconditioner build (double).
// One thread per row-segment: sum (col==row) diagonal blocks, invert 3x3.
// Atomic-free (each row owns its segment). Output invDiag column-major [N*9].
// ============================================================================
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> invDiag  : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   blocks   : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<uint>     rowIdx   : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint>     colIdx   : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<uint>     segStart : register(t3);

struct PC { uint numSeg; };
[[vk::push_constant]] PC pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.numSeg) return;

    uint start = segStart[s];
    uint end   = segStart[s + 1];
    uint row   = rowIdx[start];

    double d[9];
    [unroll] for (uint i = 0; i < 9; ++i) d[i] = 0.0;

    for (uint k = start; k < end; ++k) {
        if (colIdx[k] == row) {
            double3x3 block = load_double3x3_col_major(blocks, k);
            [unroll]
            for (uint column = 0u; column < 3u; ++column)
                [unroll]
                for (uint component = 0u; component < 3u; ++component)
                    d[column * 3u + component] += block[component][column];
        }
    }

    // A_{r,c} = d[c*3 + r]
    double a00 = d[0], a10 = d[1], a20 = d[2];
    double a01 = d[3], a11 = d[4], a21 = d[5];
    double a02 = d[6], a12 = d[7], a22 = d[8];

    double c00 = a11 * a22 - a12 * a21;
    double c01 = a12 * a20 - a10 * a22;
    double c02 = a10 * a21 - a11 * a20;
    double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (abs(det) < 1e-300) det = 1.0;
    double invDet = 1.0 / det;

    // inverse_{r,c} = cofactor_{c,r} / det
    double inv00 = c00 * invDet;
    double inv01 = (a02 * a21 - a01 * a22) * invDet;
    double inv02 = (a01 * a12 - a02 * a11) * invDet;
    double inv10 = c01 * invDet;
    double inv11 = (a00 * a22 - a02 * a20) * invDet;
    double inv12 = (a02 * a10 - a00 * a12) * invDet;
    double inv20 = c02 * invDet;
    double inv21 = (a01 * a20 - a00 * a21) * invDet;
    double inv22 = (a00 * a11 - a01 * a10) * invDet;

    double3x3 inverse;
    inverse[0][0] = inv00; inverse[1][0] = inv10; inverse[2][0] = inv20;
    inverse[0][1] = inv01; inverse[1][1] = inv11; inverse[2][1] = inv21;
    inverse[0][2] = inv02; inverse[1][2] = inv12; inverse[2][2] = inv22;
    store_double3x3_col_major(invDiag, row, inverse);
}
