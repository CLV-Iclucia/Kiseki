//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#include <Core/animation.h>
#include <Core/error.h>
#include <Core/timer.h>
#include <Core/rand-gen.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <FluidSim/fluid-types.h>
#include <FluidSim/fluid-legacy.h>
#include <FluidSim/cpu/sdf.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <memory>

namespace fluid::cpu {
using spatify::Grid;
using spatify::CellCentredGrid;
using spatify::FaceCentredGrid;
using spatify::PaddedCellCentredGrid;

class FluidSimulator final : public FluidComputeBackend {
 public:
  core::Timer timer;
  FluidSimulator(int n, const Vec3d &size, const Vec3i &resolution)
      : nParticles(n), uw(resolution + Vec3i(1, 0, 0)),
        vw(resolution + Vec3i(0, 1, 0)), ww(resolution + Vec3i(0, 0, 1)),
        pg(resolution) {
    m_particles.positions.resize(n);
    // initialize grids
    ug = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 0>>(resolution, size);
    vg = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 1>>(resolution, size);
    wg = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 2>>(resolution, size);
    ubuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 0>>(
        resolution, size);
    vbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 1>>(
        resolution, size);
    wbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 2>>(
        resolution, size);
    uValid = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
    vValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
    wValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
    uValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
    vValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
    wValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
    sdfValid = std::make_unique<Array3D<char>>(resolution);
    sdfValidBuf = std::make_unique<Array3D<char>>(resolution);
    fluidSurface = std::make_unique<SDF<3>>(resolution, size);
    fluidSurfaceBuf = std::make_unique<SDF<3>>(resolution, size);
    colliderSdf = std::make_unique<SDF<3>>(resolution, size);
    assert(
        ug->gridSpacing() == vg->gridSpacing() && vg->gridSpacing() == wg->
            gridSpacing());
  }

  void setCollider(const Mesh &colliderMesh) const override {
    std::cout << "Building collider SDF..." << std::endl;
    Array3D<int> closest(colliderSdf->width(), colliderSdf->height(),
                         colliderSdf->depth());
    Array3D<int> intersection_cnt(colliderSdf->width(), colliderSdf->height(),
                                  colliderSdf->depth());
    manifold2SDF(3, closest, intersection_cnt, colliderMesh,
                 colliderSdf.get());
    std::cout << "Done." << std::endl;
  }

  void setInitialFluid(const Mesh &fluid_mesh) override {
    for (auto &p : m_particles.positions) {
      p = core::randomVec<Real, 3>() * Vec3d(0.5, 0.5, 0.5) + Vec3d(
          0.5, 0.0, 0.5);
    }
  }

  void setAdvector(AdvectionMethod advection_method) override {
    if (advection_method == AdvectionMethod::PIC) {
      advector = std::make_unique<PicAdvector3D>(nParticles, Vec3i(pg.width(),
                                                                   pg.height(),
                                                                   pg.depth()),
                                                 pg.width() * ug->gridSpacing().x,
                                                 pg.height() * ug->gridSpacing().y,
                                                 pg.depth() * ug->gridSpacing().z);
    } else if (advection_method == AdvectionMethod::FLIP) {
      advector = std::make_unique<FlipAdvectionSolver3D>(nParticles,
                                                         Vec3i(pg.width(),
                                                               pg.height(),
                                                               pg.depth()),
                                                         pg.width() * ug->gridSpacing().x,
                                                         pg.height() * ug->gridSpacing().y,
                                                         pg.depth() * ug->gridSpacing().z);
    } else
      SIM_THROW("Unknown advection solver");
  }

  void setProjector(ProjectSolver project_solver) override {
    if (project_solver == ProjectSolver::FVM) {
      projector = std::make_unique<FvmSolver3D>(pg.width(), pg.height(),
                                                pg.depth());
      return;
    }
    SIM_THROW("Unknown projection solver");
  }

  void setCompressedSolver(CompressedSolverMethod solver_method,
                           PreconditionerMethod
                           preconditioner_method) override {
    projector->setCompressedSolver(solver_method, preconditioner_method);
  }

  void reconstruct() {
    fluidSurfaceReconstructor->reconstruct(
        m_particles.positions, 1.2 * ug->gridSpacing().x / std::sqrt(2.0),
        *fluidSurface, *sdfValid);
  }

  void smoothFluidSurface(int iters);

  void extrapolateFluidSdf(int iters) {
    for (int iter = 0; iter < iters; iter++) {
      sdfValidBuf->fill(false);
      fluidSurface->grid.forEach([&](int i, int j, int k) {
        if (sdfValid->at(i, j, k)) {
          sdfValidBuf->at(i, j, k) = true;
          return;
        }
        Real sum{0.0};
        int count{0};
        if (i > 0 && sdfValid->at(i - 1, j, k)) {
          sum += fluidSurface->grid(i - 1, j, k);
          count++;
        }
        if (i < fluidSurface->width() - 1 && sdfValid->at(i + 1, j, k)) {
          sum += fluidSurface->grid(i + 1, j, k);
          count++;
        }
        if (j > 0 && sdfValid->at(i, j - 1, k)) {
          sum += fluidSurface->grid(i, j - 1, k);
          count++;
        }
        if (j < fluidSurface->height() - 1 && sdfValid->at(i, j + 1, k)) {
          sum += fluidSurface->grid(i, j + 1, k);
          count++;
        }
        if (k > 0 && sdfValid->at(i, j, k - 1)) {
          sum += fluidSurface->grid(i, j, k - 1);
          count++;
        }
        if (k < fluidSurface->depth() - 1 && sdfValid->at(i, j, k + 1)) {
          sum += fluidSurface->grid(i, j, k + 1);
          count++;
        }
        if (count > 0) {
          fluidSurfaceBuf->grid(i, j, k) = sum / count;
          sdfValidBuf->at(i, j, k) = true;
        } else
          sdfValidBuf->at(i, j, k) = false;
      });
      std::swap(fluidSurface, fluidSurfaceBuf);
      std::swap(sdfValid, sdfValidBuf);
    }
  }

  void setReconstructor(ReconstructorMethod reconstructor_method) override {
    Vector<Real, 3>
        size(pg.width() * ug->gridSpacing().x, pg.height() * ug->gridSpacing().y, pg.depth() * ug->gridSpacing().z);
    if (reconstructor_method == ReconstructorMethod::Naive) {
      fluidSurfaceReconstructor =
          std::make_unique<NaiveReconstructor<Real, 3>>(nParticles,
                                                        pg.width(),
                                                        pg.height(),
                                                        pg.depth(),
                                                        size);
    }
  }

  void step(core::Frame &frame);

  [[nodiscard]] const std::vector<Vec3d> &positions() const {
    return m_particles.positions;
  }

  [[nodiscard]] const SDF<3> &exportFluidSurface() const {
    return *fluidSurface;
  }

  [[nodiscard]] const SDF<3> &exportColliderSurface() const {
    return *colliderSdf;
  }

  std::vector<Vec3d> &positions() { return m_particles.positions; }

 private:
  int nParticles{};

  struct ParticleList {
    std::vector<Vec3d> positions{};
  } m_particles;

  template<typename GridType>
  void extrapolate(std::unique_ptr<GridType> &g,
                   std::unique_ptr<GridType> &gbuf,
                   std::unique_ptr<Array3D<char>> &valid,
                   std::unique_ptr<Array3D<char>> &validBuf, int iters) {
    for (int iter = 0; iter < iters; iter++) {
      validBuf->fill(false);
      g->parallelForEach([&](int i, int j, int k) {
        if (valid->at(i, j, k)) {
          validBuf->at(i, j, k) = true;
          return;
        }
        Real sum{0.0};
        int count{0};
        if (i > 0 && valid->at(i - 1, j, k)) {
          sum += g->at(i - 1, j, k);
          count++;
        }
        if (i < g->width() - 1 && valid->at(i + 1, j, k)) {
          sum += g->at(i + 1, j, k);
          count++;
        }
        if (j > 0 && valid->at(i, j - 1, k)) {
          sum += g->at(i, j - 1, k);
          count++;
        }
        if (j < g->height() - 1 && valid->at(i, j + 1, k)) {
          sum += g->at(i, j + 1, k);
          count++;
        }
        if (k > 0 && valid->at(i, j, k - 1)) {
          sum += g->at(i, j, k - 1);
          count++;
        }
        if (k < g->depth() - 1 && valid->at(i, j, k + 1)) {
          sum += g->at(i, j, k + 1);
          count++;
        }
        if (count > 0) {
          gbuf->at(i, j, k) = sum / count;
          validBuf->at(i, j, k) = true;
        } else {
          gbuf->at(i, j, k) = 0.0;
          validBuf->at(i, j, k) = false;
        }
      });
      std::swap(g, gbuf);
      std::swap(valid, validBuf);
    }
  }

  void clear();
  void applyForce(Real dt) const;
  void applyCollider() const;
  void applyDirichletBoundary() const;
  [[nodiscard]] Real CFL() const;
  void substep(Real dt);
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 0>> ug, ubuf;
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 1>> vg, vbuf;
  std::unique_ptr<FaceCentredGrid<Real, Real, 3, 2>> wg, wbuf;
  std::unique_ptr<Array3D<char>> uValid, vValid, wValid, uValidBuf, vValidBuf,
      wValidBuf, sdfValid, sdfValidBuf;
  Array3D<Real> uw, vw, ww;
  Array3D<Real> pg;
  std::unique_ptr<ParticleSystemReconstructor<Real, 3>> fluidSurfaceReconstructor{};
  std::unique_ptr<SDF<3>> fluidSurface{}, fluidSurfaceBuf{}, colliderSdf{};
  std::unique_ptr<ProjectionSolver3D> projector{};
  std::unique_ptr<HybridAdvectionSolver3D> advector{};
};
} // namespace fluid::cpu
#endif  // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_