//
// Created by creeper on 10/27/23.
//

#ifndef KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#define KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#include <Spatify/arrays.h>
#include <FluidSim/cpu/sdf.h>

namespace fluid::cpu {
class HybridAdvectionSolver3D {
 public:
  HybridAdvectionSolver3D(int n, const Vec3i& resolution, Real w, Real h, Real d)
      : n_particles(n), width(w), height(h), depth(d) {
  }
  virtual void solveG2P(std::span<Vec3d> pos,
                        const FaceCentredGrid<Real, Real, 3, 0> &ug,
                        const FaceCentredGrid<Real, Real, 3, 1> &vg,
                        const FaceCentredGrid<Real, Real, 3, 2> &wg,
                        const SDF<3> &collider_sdf,
                        Real dt) = 0;
  virtual void solveP2G(std::span<Vec3d> pos,
                        FaceCentredGrid<Real, Real, 3, 0> &ug,
                        FaceCentredGrid<Real, Real, 3, 1> &vg,
                        FaceCentredGrid<Real, Real, 3, 2> &wg,
                        const SDF<3> &collider_sdf,
                        Array3D<Real> &uw,
                        Array3D<Real> &vw,
                        Array3D<Real> &ww,
                        Array3D<char> &uValid,
                        Array3D<char> &vValid,
                        Array3D<char> &wValid,
                        Real dt) = 0;
  virtual void advect(std::span<Vec3d> pos,
                      const FaceCentredGrid<Real, Real, 3, 0> &ug,
                      const FaceCentredGrid<Real, Real, 3, 1> &vg,
                      const FaceCentredGrid<Real, Real, 3, 2> &wg,
                      const SDF<3> &collider_sdf,
                      Real dt) = 0;

  [[nodiscard]] const std::vector<Vec3d>& velocities() const { return velocities_; }

  virtual ~HybridAdvectionSolver3D() = default;

 protected:

  void handleCollision(const SDF<3> &collider_sdf, Vec3d &p, Vec3d &v) const;
  [[nodiscard]] const Vec3d &vel(int i) const { return velocities_[i]; }
  Vec3d &vel(int i) { return velocities_[i]; }
  std::vector<Vec3d> velocities_{};
  int n_particles;
  Vec3i resolution{};
  Real width{};
  Real height{};
  Real depth{};
};

class PicAdvector3D final : public HybridAdvectionSolver3D {
 public:
  PicAdvector3D(int n, const Vec3i& resolution, Real width, Real height, Real depth)
      : HybridAdvectionSolver3D(n, resolution, width, height, depth) {
    velocities_.resize(n);
  }
  void solveG2P(std::span<Vec3d> pos,
                const FaceCentredGrid<Real, Real, 3, 0> &ug,
                const FaceCentredGrid<Real, Real, 3, 1> &vg,
                const FaceCentredGrid<Real, Real, 3, 2> &wg,
                const SDF<3> &collider_sdf,
                Real dt) override;
  void solveP2G(std::span<Vec3d> pos,
                FaceCentredGrid<Real, Real, 3, 0> &ug,
                FaceCentredGrid<Real, Real, 3, 1> &vg,
                FaceCentredGrid<Real, Real, 3, 2> &wg,
                const SDF<3> &collider_sdf,
                Array3D<Real> &uw,
                Array3D<Real> &vw,
                Array3D<Real> &ww,
                Array3D<char> &uValid,
                Array3D<char> &vValid,
                Array3D<char> &wValid,
                Real dt) override;
  void advect(std::span<Vec3d> pos,
              const FaceCentredGrid<Real, Real, 3, 0> &ug,
              const FaceCentredGrid<Real, Real, 3, 1> &vg,
              const FaceCentredGrid<Real, Real, 3, 2> &wg,
              const SDF<3> &collider_sdf,
              Real dt) override;

 private:
  std::unique_ptr<Array3D<Real>> uw{}, vw{}, ww{};
};
class FlipAdvectionSolver3D final : public HybridAdvectionSolver3D {
 public:
  FlipAdvectionSolver3D(int n, const Vec3i& resolution, Real width, Real height, Real depth)
      : HybridAdvectionSolver3D(n, resolution, width, height, depth) {
    velocities_.resize(n);
    u_last = std::make_unique<FaceCentredGrid<Real, Real, 3, 0>>(resolution, Vec3d(width, height, depth));
    v_last = std::make_unique<FaceCentredGrid<Real, Real, 3, 1>>(resolution, Vec3d(width, height, depth));
    w_last = std::make_unique<FaceCentredGrid<Real, Real, 3, 2>>(resolution, Vec3d(width, height, depth));
  }
  void solveG2P(std::span<Vec3d> pos,
                const FaceCentredGrid<Real, Real, 3, 0> &ug,
                const FaceCentredGrid<Real, Real, 3, 1> &vg,
                const FaceCentredGrid<Real, Real, 3, 2> &wg,
                const SDF<3> &collider_sdf,
                Real dt) override;
  void solveP2G(std::span<Vec3d> pos,
                FaceCentredGrid<Real, Real, 3, 0> &ug,
                FaceCentredGrid<Real, Real, 3, 1> &vg,
                FaceCentredGrid<Real, Real, 3, 2> &wg,
                const SDF<3> &collider_sdf,
                Array3D<Real> &uw,
                Array3D<Real> &vw,
                Array3D<Real> &ww,
                Array3D<char> &uValid,
                Array3D<char> &vValid,
                Array3D<char> &wValid,
                Real dt) override;
  void advect(std::span<Vec3d> pos,
              const FaceCentredGrid<Real, Real, 3, 0> &ug,
              const FaceCentredGrid<Real, Real, 3, 1> &vg,
              const FaceCentredGrid<Real, Real, 3, 2> &wg,
              const SDF<3> &collider_sdf,
              Real dt) override;

 private:
  std::unique_ptr<Array3D<Real>> uw{}, vw{}, ww{};
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 0>> u_last{};
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 1>> v_last{};
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 2>> w_last{};
  Real alpha = 0.97;
};
}
#endif //KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_