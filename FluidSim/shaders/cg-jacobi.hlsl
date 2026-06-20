// ============================================================================
// cg-jacobi.hlsl — CG: Jacobi preconditioner z[i] = r[i] / Adiag[i] (float32)
// ============================================================================

[[vk::binding(0, 0)]] RWStructuredBuffer<float> z      : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<float>   r      : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<float>   Adiag  : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<uint>    active : register(t2);

struct PushParams {
    uint numCells;
};
[[vk::push_constant]] PushParams pc;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.numCells) return;

    if (active[idx] == 0u) {
        z[idx] = 0.0f;
        return;
    }

    float diag = Adiag[idx];
    z[idx] = (diag > 1e-7f) ? r[idx] / diag : r[idx];
}
