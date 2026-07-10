//
// Created by creeper on 23-8-9.
//

#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_MATH_UTILS_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_MATH_UTILS_H_
#include <HairSim/hair-sim.h>
namespace hairsim {
static constexpr Real PI = 3.14159265358979323846264338327950288419716939937510;
inline Mat3d tensorProduct(const Vec3d &a, const Vec3d &b) {
  Mat3d res;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      res(i, j) = a(i) * b(j);
  return res;
}
inline Vec3d rotate(const Vec3d &axis, const Vec3d &v, Real theta) {
  return v * cos(theta) + axis.cross(v) * sin(theta) +
         axis * axis.dot(v) * (1 - cos(theta));
}
inline Mat3d skewt(const Vec3d &v) {
  Mat3d res;
  res << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return res;
}
inline Vec3d parallelTransport(const Vec3d &e0, const Vec3d &e1,
                               const Vec3d &vec) {
  Vec3d b = 2.0 * e0.cross(e1) / (e0.norm() * e1.norm() + e0.dot(e1));
  Real theta = atan(b.norm());
  return rotate(b.normalized(), vec, theta);
}
inline Real sqr(Real x) { return x * x; }
inline Real cube(Real x) { return x * x * x; }
} // namespace hairsim
#endif // KISEKI_HAIRSIM_INCLUDE_HAIRSIM_MATH_UTILS_H_
