//
// Created by creeper on 10/37/33.
//

#ifndef KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_
#define KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_
#include <Spatify/arrays.h>
#include <Spatify/grids.h>
#include <FluidSim/cpu/sdf.h>
#include <FluidSim/fluid-types.h>
#include <Core/error.h>
#include <array>
#include <memory>

namespace fluid::cpu {
using spatify::Array3D;
using spatify::GhostArray3D;
class CompressedSolver3D;

class ProjectionSolver3D {
public:
  ProjectionSolver3D() = default;

  ProjectionSolver3D(int w, int h, int d)
    : width(w), height(h), depth(d) {
  }

  virtual void buildSystem(
      const FaceCentredGrid<Real, Real, 3, 0>& ug,
      const FaceCentredGrid<Real, Real, 3, 1>& vg,
      const FaceCentredGrid<Real, Real, 3, 2>& wg,
      const SDF<3>& fluid_sdf,
      const SDF<3>& collider_sdf,
      Real dt) = 0;
  virtual Real solvePressure(const SDF<3>& fluidSdf,
                             Array3D<Real>& pg) = 0;
  virtual void project(FaceCentredGrid<Real, Real, 3, 0>& ug,
                       FaceCentredGrid<Real, Real, 3, 1>& vg,
                       FaceCentredGrid<Real, Real, 3, 2>& wg,
                       const Array3D<Real>& pg,
                       const SDF<3>& fluidSdf,
                       const SDF<3>& colliderSdf,
                       Real dt) = 0;
  virtual void setCompressedSolver(CompressedSolverMethod solver_method,
                                   PreconditionerMethod precond_method) = 0;
  virtual ~ProjectionSolver3D() = default;

protected:
  int width = 0, height = 0, depth = 0;
};

class CompressedSolver3D {
public:
  CompressedSolver3D(int w, int h, int d)
    : r(w, h, d) {
  }

  virtual Real solve(const Array3D<Real>& Adiag,
                     const std::array<Array3D<Real>, 6>& Aneighbour,
                     const Array3D<Real>& rhs,
                     const Array3D<uint8_t>& active,
                     Array3D<Real>& pg) = 0;
  virtual void setPreconditioner(PreconditionerMethod precond_method) = 0;
  virtual ~CompressedSolver3D() = default;

protected:
  Array3D<Real> r;
};

class Preconditioner3D {
public:
  virtual void precond(
      const Array3D<Real>& Adiag,
      const std::array<Array3D<Real>, 6>& Aneighbour,
      const Array3D<Real>& r, const Array3D<uint8_t>& active,
      Array3D<Real>& z) = 0;
  virtual ~Preconditioner3D() = default;
};

class ModifiedIncompleteCholesky3D final : public Preconditioner3D {
public:
  ModifiedIncompleteCholesky3D(int w, int h, int d, Real tuning = 0.97,
                               Real safety = 0.25)
    : E(w, h, d), q(w, h, d), tau(tuning), sigma(safety) {
  }

  void precond(
      const Array3D<Real>& Adiag,
      const std::array<Array3D<Real>, 6>& Aneighbour,
      const Array3D<Real>& r, const Array3D<uint8_t>& active,
      Array3D<Real>& z) override;

protected:
  Array3D<Real> E;
  Array3D<Real> q;
  Real tau, sigma;
};

class CgSolver3D final : public CompressedSolver3D {
public:
  CgSolver3D(int w, int h, int d)
    : CompressedSolver3D(w, h, d), z(w, h, d),
      s(w, h, d) {
  }

  Real solve(const Array3D<Real>& Adiag,
             const std::array<Array3D<Real>, 6>& Aneighbour,
             const Array3D<Real>& rhs,
             const Array3D<uint8_t>& active,
             Array3D<Real>& pg) override;

  void setPreconditioner(PreconditionerMethod precond_method) override {
    if (precond_method == PreconditionerMethod::ModifiedIncompleteCholesky) {
      preconditioner = std::make_unique<ModifiedIncompleteCholesky3D>(
          r.width(), r.height(), r.depth());
    } else if (precond_method == PreconditionerMethod::Multigrid) {
      SIM_THROW("Multigrid preconditioner not supported yet");
    } else
      SIM_THROW("Unknown preconditioner type");
  }

  ~CgSolver3D() override = default;

protected:
  Array3D<Real> z;
  Array3D<Real> s;
  std::unique_ptr<Preconditioner3D> preconditioner = {};
  Real tolerance = 1e-6;
  int max_iterations = 300;
};

class FvmSolver3D final : public ProjectionSolver3D {
public:
  FvmSolver3D(int w, int h, int d)
    : ProjectionSolver3D(w, h, d), Adiag(w, h, d), rhs(w, h, d), uWeights(w + 1, h, d),
      vWeights(w, h + 1, d), wWeights(w, h, d + 1),
      Aneighbour{Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),
                 Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),
                 Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),},
      active(w, h, d) {
  }

  void buildSystem(
      const FaceCentredGrid<Real, Real, 3, 0>& ug,
      const FaceCentredGrid<Real, Real, 3, 1>& vg,
      const FaceCentredGrid<Real, Real, 3, 2>& wg,
      const SDF<3>& fluidSdf,
      const SDF<3>& colliderSdf,
      Real dt) override;

  Real solvePressure(const SDF<3>& fluidSdf, Array3D<Real>& pg) override {
    return solver->solve(Adiag, Aneighbour, rhs, active, pg);
  }

  void project(FaceCentredGrid<Real, Real, 3, 0>& ug,
               FaceCentredGrid<Real, Real, 3, 1>& vg,
               FaceCentredGrid<Real, Real, 3, 2>& wg,
               const Array3D<Real>& pg,
               const SDF<3>& fluidSdf,
               const SDF<3>& colliderSdf,
               Real dt) override;

  void setCompressedSolver(CompressedSolverMethod solver_method,
                           PreconditionerMethod precond_method) override {
    if (solver_method == CompressedSolverMethod::CG) {
      auto cg_solver = std::make_unique<CgSolver3D>(width, height, depth);
      cg_solver->setPreconditioner(precond_method);
      solver = std::move(cg_solver);
    } else
      SIM_THROW("Unknown solver type");
  }

  ~FvmSolver3D() override = default;

protected:
  void checkSystemSymmetry() const;
  Array3D<Real> Adiag, rhs, uWeights, vWeights, wWeights;
  std::array<Array3D<Real>, 6> Aneighbour;
  std::unique_ptr<CompressedSolver3D> solver{};
  Array3D<uint8_t> active;
};
}
#endif //KISEKI_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_