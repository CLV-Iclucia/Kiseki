// ============================================================================
// include/FluidSim/kernel.h
// Cubic B-spline kernel functions for SPH / FLIP transfer
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <cmath>

namespace fluid {

// ---- Cubic B-spline 1D basis (SPH kernel building block) ----
inline Real cubicBSpline1D(Real x) {
    x = std::abs(x);
    if (x < 1.0) return 0.5 * x * x * x - x * x + 2.0 / 3.0;
    if (x < 2.0) { Real t = 2.0 - x; return t * t * t / 6.0; }
    return 0.0;
}

// Tensor-product cubic B-spline weight in 3D: w(delta, h) = ∏ B(delta_i / h)
inline Real cubicWeight3D(Real h, const Vec3d& delta) {
    return cubicBSpline1D(delta.x / h)
         * cubicBSpline1D(delta.y / h)
         * cubicBSpline1D(delta.z / h);
}

// 2D version (used by legacy 2D code paths)
inline Real cubicWeight2D(Real h, const Vec2d& delta) {
    return cubicBSpline1D(delta.x / h)
         * cubicBSpline1D(delta.y / h);
}

} // namespace fluid
