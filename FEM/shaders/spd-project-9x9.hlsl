// ============================================================================
// spd-project-9x9.hlsl — batch SPD filter of symmetric 9x9 matrices.
// One thread per matrix. Matrices stored row-major: M[i][j] = buf[m*81+i*9+j].
// Strategy/clamp selected by defines forwarded to jacobi-eigen.hlsli.
// ============================================================================
#include "jacobi-eigen.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> matsOut : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<double>   matsIn  : register(t0);

struct PC { uint count; };
[[vk::push_constant]] PC pc;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint m = tid.x;
    if (m >= pc.count) return;

    uint base = m * 81u;
    double H[9][9];
    [unroll] for (int i = 0; i < 9; ++i)
        [unroll] for (int j = 0; j < 9; ++j)
            H[i][j] = matsIn[base + i * 9 + j];

    filterSymmetric9(H);

    [unroll] for (int i = 0; i < 9; ++i)
        [unroll] for (int j = 0; j < 9; ++j)
            matsOut[base + i * 9 + j] = H[i][j];
}
