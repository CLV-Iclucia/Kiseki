// ============================================================================
// test-ee-mollifier.cc — validate the shared (C++/HLSL) edge-edge mollifier
// <fem/shared/gipc-mollifier.h> against the GIPC CPU reference
// fem::ipc::gipc::computeMollifiedBarrier (+ computePFPx_PEE).
//
// For several near-parallel edge pairs (I1 < eps_x, distance < dHat) the shared
// mollified gradient (4 dvec3) and rank-<=2 Hessian (expanded from its w/lam
// generators) must match the reference gradient/Hessian. Pure CPU test (no GPU)
// — it pins down the maths port that both the CPU unified barrier path and the
// GPU barrier assembler now consume.
//
// Edges are oriented with eb0.y > eb1.y so the reference computePFPx_PEE autogen
// (which carries a spurious debug assert t = eb0.y-eb1.y > 0) does not abort.
// ============================================================================
#include <fem/shared/gipc-mollifier.h>
#include <fem/ipc/gipc/mollifier.h>
#include <fem/ipc/gipc/pfpx.h>
#include <fem/ipc/gipc/barrier.h>
#include <fem/ipc/distances.h>

#include <gtest/gtest.h>
#include <glm/glm.hpp>

#include <array>
#include <vector>

using namespace sim;
namespace shared = sim::fem::shared;
namespace gipc = sim::fem::ipc::gipc;

namespace {

struct EdgeCase {
    glm::dvec3 ea0, ea1, eb0, eb1;
};

// Build a near-parallel edge pair: edge A along +x; edge B nearly parallel,
// offset by height h, with a small skew (delta, eps) so the cross product is
// tiny (I1 < eps_x). eb0.y > eb1.y (delta > 0) to dodge the reference assert.
EdgeCase makeNearParallel(double h, double delta, double eps, double xoff) {
    EdgeCase c;
    c.ea0 = {0.0, 0.0, 0.0};
    c.ea1 = {1.0, 0.0, 0.0};
    c.eb0 = {xoff,        h + delta, 0.0};
    c.eb1 = {xoff + 1.0,  h - delta, eps};
    return c;
}

double maxAbs(const glm::dvec3& v) {
    return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
}

}  // namespace

TEST(EEMollifier, MatchesGipcReference) {
    const double dHat = 1.0, kappa = 1e3;
    gipc::Barrier barrier(dHat);

    std::vector<EdgeCase> cases = {
        makeNearParallel(0.10, 0.010, 0.010, 0.05),
        makeNearParallel(0.20, 0.005, 0.015, -0.10),
        makeNearParallel(0.05, 0.012, 0.006, 0.20),
        makeNearParallel(0.15, 0.008, 0.012, 0.00),
    };

    int validated = 0;
    for (const auto& c : cases) {
        // rest == current (eps_x derived from these edge lengths).
        const glm::dvec3 &ra0 = c.ea0, &ra1 = c.ea1, &rb0 = c.eb0, &rb1 = c.eb1;

        // Sanity: this config must actually trigger the mollifier branch.
        ASSERT_TRUE(shared::shEEUsesMollifier(c.ea0, c.ea1, c.eb0, c.eb1,
                                              ra0, ra1, rb0, rb1))
            << "case is not near-parallel enough";

        double dSqr = fem::ipc::distanceSqrLineLine(c.ea0, c.ea1, c.eb0, c.eb1);
        ASSERT_LT(dSqr, barrier.dHatSqr()) << "case not active";

        // ---- Reference (GIPC CPU) ----
        auto pfpx = gipc::computePFPx_PEE(c.ea0, c.ea1, c.eb0, c.eb1, dHat);
        ASSERT_TRUE(pfpx.valid);
        auto ref = gipc::computeMollifiedBarrier(
            c.ea0, c.ea1, c.eb0, c.eb1, ra0, ra1, rb0, rb1,
            dSqr, pfpx.PFPx, barrier, kappa);
        ASSERT_TRUE(ref.active);

        // ---- Shared single-source ----
        auto m = shared::shMollifiedBarrierEE(
            c.ea0, c.ea1, c.eb0, c.eb1, ra0, ra1, rb0, rb1, dHat, kappa);
        ASSERT_TRUE(m.active);

        // ---- Gradient ----
        double gNorm = 0.0, gErr = 0.0;
        for (int v = 0; v < 4; ++v) {
            glm::dvec3 gs(m.grad[v * 3], m.grad[v * 3 + 1], m.grad[v * 3 + 2]);
            gNorm = std::max(gNorm, maxAbs(ref.gradient[v]));
            gErr  = std::max(gErr, maxAbs(gs - ref.gradient[v]));
        }
        ASSERT_GT(gNorm, 0.0);
        EXPECT_LT(gErr, 1e-8 * std::max(1.0, gNorm)) << "gradient mismatch";

        // ---- Hessian: expand H = Σ lam[k] · w[k] w[k]ᵀ and compare blocks ----
        double hNorm = 0.0, hErr = 0.0;
        for (int I = 0; I < 4; ++I)
            for (int J = 0; J < 4; ++J) {
                const glm::dmat3& refB = ref.hessian[I][J];
                for (int cc = 0; cc < 3; ++cc)
                    for (int rr = 0; rr < 3; ++rr) {
                        double hs = 0.0;
                        for (int k = 0; k < m.rank; ++k)
                            hs += m.lam[k] * m.w[k][I * 3 + rr] * m.w[k][J * 3 + cc];
                        double hr = refB[cc][rr];  // glm dmat3 col-major: [col][row]
                        hNorm = std::max(hNorm, std::abs(hr));
                        hErr  = std::max(hErr, std::abs(hs - hr));
                    }
            }
        EXPECT_LT(hErr, 1e-8 * std::max(1.0, hNorm)) << "hessian mismatch";
        ++validated;
    }
    EXPECT_EQ(validated, int(cases.size()));
}

// The mollifier must ramp to 0 as the edges align (energy continuity): when
// I1 -> 0 the mollified contribution vanishes (m_eps(I1) -> 0).
TEST(EEMollifier, VanishesAtPerfectAlignment) {
    const double dHat = 1.0, kappa = 1e3;
    // Exactly parallel edges (cross == 0): no contribution.
    glm::dvec3 ea0(0, 0, 0), ea1(1, 0, 0);
    glm::dvec3 eb0(0.0, 0.1, 0.0), eb1(1.0, 0.1, 0.0);
    auto m = shared::shMollifiedBarrierEE(ea0, ea1, eb0, eb1,
                                          ea0, ea1, eb0, eb1, dHat, kappa);
    EXPECT_FALSE(m.active);  // I1 == 0 -> inactive (singular, skipped)

    double e = shared::shMollifiedEnergyEE(ea0, ea1, eb0, eb1,
                                           ea0, ea1, eb0, eb1, dHat, kappa);
    EXPECT_EQ(e, 0.0);
}
