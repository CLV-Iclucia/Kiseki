// ============================================================================
// barrier-assemble.hlsl — module 2, GIPC barrier gradient + Hessian assembly.
// One thread per active ConstraintPair (grouped PP|PE|PT|EE per typeOffsets).
// Computes the shared contact vector v + I5, then emits:
//   gradient entries : (row = vertex, vec3 = kappa*gradCoeff(I5)*sqrt(I5)*v_bi)
//   Hessian blocks   : (row,col, 3x3 = kappa*clampedLambda0(I5)*v_bi (x) v_bj)
// Write offsets are derived from the per-kind bucket bases (no atomics, no scan)
// since the pairs are homogeneous within each kind bucket.
// Mirrors CPU constraintPairBarrier{Gradient,Hessian}, including the near-parallel
// edge-edge mollifier (shared <fem/shared/gipc-mollifier.h>): EE pairs with
// I1 < eps_x route to shMollifiedBarrierEE (still 4 grad entries + 16 blocks,
// same layout), restoring C1 continuity of the contact barrier.
// ============================================================================
#include <fem/shared/gipc-pfpx.h>
#include <fem/shared/gipc-barrier.h>
#include <fem/shared/gipc-mollifier.h>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessBlocks : register(u0); // [Hn*9] col-major dmat3
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   hessRow    : register(u1); // [Hn]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   hessCol    : register(u2); // [Hn]
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>   gradRow    : register(u3); // [Gn]
[[vk::binding(4, 0)]] RWStructuredBuffer<double> gradVal    : register(u4); // [Gn*3]
[[vk::binding(5, 0)]] StructuredBuffer<int>      pairs      : register(t0); // [total*4]
[[vk::binding(6, 0)]] StructuredBuffer<double>   x          : register(t1); // [nVerts*3]
[[vk::binding(7, 0)]] StructuredBuffer<uint>     uParams    : register(t2); // typeOffsets[5], hessBase[4], gradBase[4]
[[vk::binding(8, 0)]] StructuredBuffer<double>   dParams    : register(t3); // [kappa, dHat]
[[vk::binding(9, 0)]] StructuredBuffer<double>   xRest      : register(t4); // [nVerts*3] rest config (EE mollifier eps_x)

struct PC { uint total; };
[[vk::push_constant]] PC pc;

double3 loadP(int i) { return load_double3(x, uint(i)); }
double3 loadR(int i) { return load_double3(xRest, uint(i)); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint s = tid.x;
    if (s >= pc.total) return;

    // ---- locate kind bucket ----
    uint to1 = uParams[1], to2 = uParams[2], to3 = uParams[3];
    int k;
    if      (s < to1) k = 0;
    else if (s < to2) k = 1;
    else if (s < to3) k = 2;
    else              k = 3;
    int  vertsN = (k == 0) ? 2 : ((k == 1) ? 3 : 4);
    uint base   = uParams[k];          // typeOffsets[k]
    uint hBase  = uParams[5 + k];
    uint gBase  = uParams[9 + k];
    uint local  = s - base;
    uint hOff   = hBase + local * uint(vertsN * vertsN);
    uint gOff   = gBase + local * uint(vertsN);

    int idx[4];
    idx[0] = pairs[s * 4 + 0]; idx[1] = pairs[s * 4 + 1];
    idx[2] = pairs[s * 4 + 2]; idx[3] = pairs[s * 4 + 3];

    double kappa = dParams[0], dHat = dParams[1], dHatSqr = dHat * dHat;

    double3 p0 = loadP(idx[0]);
    double3 p1 = loadP(idx[1]);
    double3 p2 = loadP(max(idx[2], 0));  // unused (-1) for PP -> harmless x[0]
    double3 p3 = loadP(max(idx[3], 0));  // unused (-1) for PP/PE

    // ---- near-parallel edge-edge: mollified branch (same 4-grad/16-block layout) ----
    if (k == 3) {
        double3 r0 = loadR(idx[0]);
        double3 r1 = loadR(idx[1]);
        double3 r2 = loadR(idx[2]);
        double3 r3 = loadR(idx[3]);
        if (shEEUsesMollifier(p0, p1, p2, p3, r0, r1, r2, r3)) {
            ShMollifiedEE m = shMollifiedBarrierEE(p0, p1, p2, p3, r0, r1, r2, r3, dHat, kappa);
            for (int bi = 0; bi < 4; ++bi) {
                gradRow[gOff + uint(bi)] = uint(idx[bi]);
                gradVal[(gOff + uint(bi)) * 3 + 0] = m.active ? m.grad[bi * 3 + 0] : 0.0;
                gradVal[(gOff + uint(bi)) * 3 + 1] = m.active ? m.grad[bi * 3 + 1] : 0.0;
                gradVal[(gOff + uint(bi)) * 3 + 2] = m.active ? m.grad[bi * 3 + 2] : 0.0;
            }
            for (int bi = 0; bi < 4; ++bi)
                for (int bj = 0; bj < 4; ++bj) {
                    uint bIdx = hOff + uint(bi * 4 + bj);
                    hessRow[bIdx] = uint(idx[bi]);
                    hessCol[bIdx] = uint(idx[bj]);
                    for (int c = 0; c < 3; ++c)
                        for (int rr = 0; rr < 3; ++rr) {
                            double val = 0.0;
                            if (m.active)
                                for (int kk = 0; kk < m.rank; ++kk)
                                    val += m.lam[kk] * m.w[kk][bi * 3 + rr] * m.w[kk][bj * 3 + c];
                            hessBlocks[bIdx * 9 + uint(c * 3 + rr)] = val;
                        }
                }
            return;
        }
    }

    ShGipcPfpx pf;
    if      (k == 0) pf = shGipcPFPx_PP(p0, p1, dHat);
    else if (k == 1) pf = shGipcPFPx_PE(p0, p1, p2, dHat);
    else if (k == 2) pf = shGipcPFPx_PT(p0, p1, p2, p3, dHat);
    else             pf = shGipcPFPx_EE(p0, p1, p2, p3, dHat);

    double gradScalar = kappa * shBarrierGradCoeff(pf.I5, dHatSqr) * shSqrt(pf.I5);
    double lam        = kappa * shBarrierClampedLambda0(pf.I5, dHatSqr);

    // ---- gradient entries (v is zero when inactive/degenerate -> harmless) ----
    for (int bi = 0; bi < vertsN; ++bi) {
        gradRow[gOff + uint(bi)] = uint(idx[bi]);
        gradVal[(gOff + uint(bi)) * 3 + 0] = gradScalar * pf.v[bi * 3 + 0];
        gradVal[(gOff + uint(bi)) * 3 + 1] = gradScalar * pf.v[bi * 3 + 1];
        gradVal[(gOff + uint(bi)) * 3 + 2] = gradScalar * pf.v[bi * 3 + 2];
    }

    // ---- Hessian blocks: lam * v_bi (x) v_bj, column-major 3x3 ----
    for (int bi = 0; bi < vertsN; ++bi)
        for (int bj = 0; bj < vertsN; ++bj) {
            uint bIdx = hOff + uint(bi * vertsN + bj);
            hessRow[bIdx] = uint(idx[bi]);
            hessCol[bIdx] = uint(idx[bj]);
            for (int c = 0; c < 3; ++c)
                for (int rr = 0; rr < 3; ++rr)
                    hessBlocks[bIdx * 9 + uint(c * 3 + rr)] =
                        lam * pf.v[bi * 3 + rr] * pf.v[bj * 3 + c];
        }
}
