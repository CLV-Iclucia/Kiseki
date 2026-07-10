//
// Created by creeper on 23-8-10.
//

#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_

#include <HairSim/hair.h>
#include <HairSim/utils.h>
#include <cassert>
#include <cmath>

namespace hairsim {
inline Vec3d pmpe(const Hair& hair, Index i, Index j) {
  assert(abs(i - j) <= 1);
  if (j == i - 1 || j == i)
    return 0.5 * hair.curvatureBinormal(i) / hair.edgeLength(j);
  return {};
}
inline Vec3d pInvlen_pe(const Hair& hair, Index i, Index j) {
  assert(abs(i - j) <= 1);
  if (j == i)
    return -hair.edge(j) / cube(hair.edgeLength(j));
  return {};
}
inline Vec3d pKappa1_pe(const Hair& hair, Index i, Index j, Index k) {
  assert(j == i + 1 || j == i);
  assert(j == k + 1 || j == k);
  Real sgn = j == k ? -1.0 : 1.0;
  Vec3d t_v = j == k ? hair.vertexTangent(j - 1) : hair.vertexTangent(j);
  return 1.0 / hair.edgeLength(k) *
         (-hair.kappa1(i, j) * hair.tangentTilde(j) +
          2.0 * sgn * t_v.cross(hair.m2(i)) /
          (1.0 + hair.tangent(j - 1).dot(hair.tangent(j))));
}
inline Vec3d pKappa2_pe(const Hair& hair, Index i, Index j, Index k) {
  assert(j == i + 1 || j == i);
  assert(j == k + 1 || j == k);
  Real sgn = j == k ? 1.0 : -1.0;
  Vec3d t_v = j == k ? hair.vertexTangent(j - 1) : hair.vertexTangent(j);
  return 1.0 / hair.edgeLength(k) *
         (-hair.kappa2(i, j) * hair.tangentTilde(j) +
          2.0 * sgn * t_v.cross(hair.m1(i)) /
          (1.0 + hair.tangent(j - 1).dot(hair.tangent(j))));
}
inline Mat43d pCurv_pe(const Hair& hair, Index i, Index j) {
  Mat43d res;
  res << pKappa1_pe(hair, i, i, j).transpose(),
      pKappa1_pe(hair, i + 1, i, j).transpose(),
      pKappa2_pe(hair, i, i, j).transpose(),
      pKappa2_pe(hair, i + 1, i, j).transpose();
  return res;
}
inline Real pEt_pm(const HairParams& params, const Hair& hair, Index i) {
  return params.G * params.area() / (4 * hair.vertexReferenceLength(i)) *
         params.radius * params.radius * hair.m(i);
}
inline Vec3d pEt_pe(const HairParams& params, const Hair& hair, Index i) {
  return pEt_pm(params, hair, i) * pmpe(hair, i, i) +
         pEt_pm(params, hair, i + 1) * pmpe(hair, i + 1, i);
}
inline Mat3d pkb_pe(const Hair& hair, Index i, Index j) {
  assert(j == i - 1 || j == i || j == i + 1);
  if (j == i)
    return -2.0 / hair.edgeLength(i) *
           tensorProduct(
               hair.curvatureBinormal(i) /
               (1.0 + hair.tangent(i - 1).dot(hair.tangent(i))),
               hair.vertexTangent(i)) +
           2.0 * skewt(hair.tangent(i - 1)) /
           (hair.edgeLength(i) *
            (1.0 + hair.tangent(i - 1).dot(hair.tangent(i))));
  if (j == i - 1)
    return -2.0 / hair.edgeLength(i) *
           tensorProduct(hair.curvatureBinormal(i),
                         hair.tangent(i - 1) + hair.tangent(i)) /
           (hair.edgeLength(i - 1) *
            (1.0 + hair.tangent(i - 1).dot(hair.tangent(i)))) +
           2.0 * skewt(hair.tangent(i)) /
           (hair.edgeLength(i - 1) * hair.edgeLength(i) +
            hair.edge(i - 1).dot(hair.edge(i)));
  return -pkb_pe(hair, i, i) - pkb_pe(hair, i, i - 1);
}
inline Mat3d p2m_pe2(const Hair& hair, Index i, Index j, Index k) {
  return 0.5 * pkb_pe(hair, i, j) / hair.edgeLength(k) +
         0.5 *
         tensorProduct(hair.curvatureBinormal(i), pInvlen_pe(hair, k, j));
}
} // namespace hairsim

#endif // KISEKI_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_