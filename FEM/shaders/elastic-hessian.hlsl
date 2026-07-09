// ============================================================================
// elastic-hessian.hlsl — Stable Neo-Hookean elastic Hessian (double).
//
// Per tet (one thread):
//   F = Ds*DmInv;  g = cof(F);  J = det(F)
//   H_F(9x9) = lambda*vec(g)vec(g)^T + mu*I9 + ((J-1)lambda - mu)*hessIiii(F)
//   filterSymmetric9(H_F)            (Jacobi eigen-decomp + abs clamp)
//   H_x(12x12) = PFPx^T * H_F * PFPx * vol
//   emit 16 BCOO 3x3 blocks (column-major), rowIdx/colIdx = tet vertices.
//
// hessIiii follows invariants.h (skewt = cross-product matrix [v]x).
// Block storage matches glm::dmat3 value_ptr: blocks[k*9 + col*3 + row].
// ============================================================================
#include "jacobi-eigen.hlsli"
#include <RHI/structured-buffer-access.hlsli>

[[vk::binding(0, 0)]] RWStructuredBuffer<double> outBlocks : register(u0);  // [16M*9]
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   outRow    : register(u1);  // [16M]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   outCol    : register(u2);  // [16M]
[[vk::binding(3, 0)]] StructuredBuffer<double>   x         : register(t0);  // [N*3]
[[vk::binding(4, 0)]] StructuredBuffer<uint>     tetConn   : register(t1);  // [M*4]
[[vk::binding(5, 0)]] StructuredBuffer<double>   DmInv     : register(t2);  // [M*9]
[[vk::binding(6, 0)]] StructuredBuffer<double>   vol       : register(t3);  // [M]
[[vk::binding(7, 0)]] StructuredBuffer<double>   material  : register(t4);  // [mu, lambda]

struct PC { uint numTets; };
[[vk::push_constant]] PC pc;

double3 ld(uint i)              { return load_double3(x, i); }
double3 dcross(double3 a, double3 b) { return double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
double  ddot(double3 a, double3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }

// H block(rb,cb) += sgn * skewt(v),  skewt(v)=[[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]]
void addSkew(inout double H[9][9], int rb, int cb, double sgn, double3 v) {
    H[rb + 0][cb + 1] += sgn * (-v.z);
    H[rb + 0][cb + 2] += sgn * ( v.y);
    H[rb + 1][cb + 0] += sgn * ( v.z);
    H[rb + 1][cb + 2] += sgn * (-v.x);
    H[rb + 2][cb + 0] += sgn * (-v.y);
    H[rb + 2][cb + 1] += sgn * ( v.x);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= pc.numTets) return;

    uint cb = t * 4u;
    double3x3 D = load_double3x3_col_major(DmInv, t);
    uint i0 = tetConn[cb + 0], i1 = tetConn[cb + 1], i2 = tetConn[cb + 2], i3 = tetConn[cb + 3];
    double3 p0 = ld(i0);
    double3 e0 = ld(i1) - p0, e1 = ld(i2) - p0, e2 = ld(i3) - p0;

    // F(:,c) = e0*DmInv(0,c) + e1*DmInv(1,c) + e2*DmInv(2,c)
    double F[3][3];
    for (uint c = 0; c < 3; ++c) {
        double m0 = D[0][c], m1 = D[1][c], m2 = D[2][c];
        F[0][c] = e0.x * m0 + e1.x * m1 + e2.x * m2;
        F[1][c] = e0.y * m0 + e1.y * m1 + e2.y * m2;
        F[2][c] = e0.z * m0 + e1.z * m1 + e2.z * m2;
    }

    double mu = material[0], lambda = material[1];
    double3 f0 = double3(F[0][0], F[1][0], F[2][0]);
    double3 f1 = double3(F[0][1], F[1][1], F[2][1]);
    double3 f2 = double3(F[0][2], F[1][2], F[2][2]);

    double3 g0 = dcross(f1, f2), g1 = dcross(f2, f0), g2 = dcross(f0, f1);  // cofactor cols
    double  J  = ddot(f0, g0);

    double vecg[9];
    vecg[0] = g0.x; vecg[1] = g0.y; vecg[2] = g0.z;
    vecg[3] = g1.x; vecg[4] = g1.y; vecg[5] = g1.z;
    vecg[6] = g2.x; vecg[7] = g2.y; vecg[8] = g2.z;

    double coef = (J - 1.0) * lambda - mu;

    // H_F = lambda*vecg vecg^T + mu*I9 + coef*hessIiii(F)
    double H[9][9];
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            H[i][j] = lambda * vecg[i] * vecg[j];
    for (int i = 0; i < 9; ++i) H[i][i] += mu;

    addSkew(H, 3, 0,  coef, f2);
    addSkew(H, 6, 0, -coef, f1);
    addSkew(H, 0, 3, -coef, f2);
    addSkew(H, 6, 3,  coef, f0);
    addSkew(H, 0, 6,  coef, f1);
    addSkew(H, 3, 6, -coef, f0);

    filterSymmetric9(H);  // eigen-filter (Jacobi + abs clamp)

    // PFPx (9x12), sparse: nonzero only when F-row == DOF-component.
    double B[9][12];
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 12; ++j) B[i][j] = 0.0;
    for (uint c = 0; c < 3; ++c) {
        double d0 = D[0][c], d1 = D[1][c], d2 = D[2][c];
        double cs = d0 + d1 + d2;
        for (uint d = 0; d < 3; ++d) {
            uint n = c * 3 + d;       // F-row r = d
            B[n][0 * 3 + d] = -cs;    // vertex 0
            B[n][1 * 3 + d] = d0;     // vertex 1
            B[n][2 * 3 + d] = d1;     // vertex 2
            B[n][3 * 3 + d] = d2;     // vertex 3
        }
    }

    // tmp = H * B   (9x12)
    double tmp[9][12];
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 12; ++j) {
            double s = 0.0;
            for (int k = 0; k < 9; ++k) s += H[i][k] * B[k][j];
            tmp[i][j] = s;
        }

    // H_x = B^T * tmp * vol  → emit 16 blocks
    double w = vol[t];
    uint outBase = t * 16u;
    uint conn[4] = { i0, i1, i2, i3 };
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            uint bi = outBase + a * 4 + b;
            outRow[bi] = conn[a];
            outCol[bi] = conn[b];
            for (int r = 0; r < 3; ++r)
                for (int cc = 0; cc < 3; ++cc) {
                    double s = 0.0;
                    for (int n = 0; n < 9; ++n) s += B[n][a * 3 + r] * tmp[n][b * 3 + cc];
                    outBlocks[bi * 9 + cc * 3 + r] = s * w;
                }
        }
}
