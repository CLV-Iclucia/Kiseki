//
// ipc-ccd.h — shared (C++ + HLSL) additive CCD (ACCD).
//
// Single source of truth for the per-pair additive continuous collision
// detection time-of-impact, mirroring CollisionDetector::runACCD (the
// non-reserved path used by the broad-phase candidate CCD). Each candidate is
// the 4 vertices of a VT (point p0 + triangle p1,p2,p3) or EE (edge p0,p1 +
// edge p2,p3) pair, with absolute positions `x*` and per-vertex step `u*`
// (= alpha-direction). The conservative step bound is the global min over all
// candidates' tois.
//
//   `s`   : ACCD gap fraction (CollisionDetector::s, default 0.1)
//   `toi` : current cap (the search upper bound, typically 1.0)
//   return: a toi in (0, toi] if a contact limits the step, else `toi`
//           (no limit) — so a global min reduction yields the step bound.
//
#ifndef KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_CCD_H_
#define KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_CCD_H_

#include <fem/shared/ipc-distance.h>

#define SH_CCD_EE 0
#define SH_CCD_VT 1

#define SH_ACCD_MAX_ITERS 2000

SH_NS_BEGIN
    SH_INLINE sh_real shLength(sh_real3 v) { return shSqrt(shDot(v, v)); }

    // ACCD time-of-impact for one VT/EE pair (matches CollisionDetector::runACCD).
    SH_INLINE sh_real shACCD(int mode,
                             sh_real3 x0, sh_real3 x1, sh_real3 x2, sh_real3 x3,
                             sh_real3 u0, sh_real3 u1, sh_real3 u2, sh_real3 u3,
                             sh_real toi, sh_real s)
    {
        // Mean-subtract the step (removes rigid translation; ACCD measures relative
        // approach), exactly like runACCD(const ACCDOptions&).
        sh_real3 pBar = (u0 + u1 + u2 + u3) * 0.25;
        sh_real3 xx[4];
        sh_real3 pp[4];
        xx[0] = x0;
        xx[1] = x1;
        xx[2] = x2;
        xx[3] = x3;
        pp[0] = u0 - pBar;
        pp[1] = u1 - pBar;
        pp[2] = u2 - pBar;
        pp[3] = u3 - pBar;

        sh_real lp;
        if (mode == SH_CCD_EE)
            lp = shMax(shLength(pp[0]), shLength(pp[1])) +
                shMax(shLength(pp[2]), shLength(pp[3]));
        else
            lp = shLength(pp[0]) +
                shMax(shMax(shLength(pp[1]), shLength(pp[2])), shLength(pp[3]));
        if (lp == 0.0) return toi; // no relative motion -> no limit

        sh_real dis = (mode == SH_CCD_EE)
                          ? shSqrt(shDistanceSqrEdgeEdge(xx[0], xx[1], xx[2], xx[3]))
                          : shSqrt(shDistanceSqrPointTriangle(xx[0], xx[1], xx[2], xx[3]));
        sh_real g = s * dis;
        sh_real t = 0.0;
        sh_real tl = (1.0 - s) * (dis / lp);

        for (int it = 0; it < SH_ACCD_MAX_ITERS; ++it)
        {
            for (int i = 0; i < 4; ++i) xx[i] = xx[i] + pp[i] * tl;
            dis = (mode == SH_CCD_EE)
                      ? shSqrt(shDistanceSqrEdgeEdge(xx[0], xx[1], xx[2], xx[3]))
                      : shSqrt(shDistanceSqrPointTriangle(xx[0], xx[1], xx[2], xx[3]));
            if (dis < g + 1e-10)
            {
                if (t == 0.0) t = tl;
                return t; // contact
            }
            t += tl;
            if (t > toi) return toi; // no contact within the cap
            tl = 0.9 * dis / lp;
        }
        // Hit the iteration guard: conservative fallback (matches CPU t>0 ? t : none).
        return (t > 0.0) ? t : toi;
    }

SH_NS_END

#endif  // KISEKI_FEM_INCLUDE_FEM_SHARED_IPC_CCD_H_
