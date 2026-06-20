//
// Created by creeper on 10/25/23.
//

#ifndef JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_
#define JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_
#include <FluidSim/fluid-types.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <Spatify/ns-util.h>
#include <FluidSim/cpu/util.h>
#include <fstream>

namespace core {
struct Mesh;
}

namespace fluid {
// use grids to store the SDF
template <int Dim>
struct SDF : NonCopyable {
  SDF(const Vector<int, Dim>& resolution, const Vector<Real, Dim>& size,
      const Vector<Real, Dim>& origin = Vector<Real, Dim>(0.0))
    : grid(resolution, size, origin) {
  }
  SDF(SDF&& other) noexcept : grid(std::move(other.grid)) {
  }
  void init(const Vector<int, Dim>& resolution, const Vector<Real, Dim>& size,
            const Vector<Real, Dim>& origin = Vector<Real, Dim>(0.0)) {
    grid.init(resolution, size, origin);
  }
  void clear() {
    grid.clear();
  }
  [[nodiscard]] Real eval(const Vector<Real, Dim>& p) const {
    if constexpr (Dim == 2) {
    } else if constexpr (Dim == 3) {
      Vec3i idx = grid.coordToSampledCellIndex(p);
      int x = clamp(idx.x, 0, grid.width() - 1);
      int y = clamp(idx.y, 0, grid.height() - 1);
      int z = clamp(idx.z, 0, grid.depth() - 1);
      int xn = clamp(idx.x + 1, 0, grid.width() - 1);
      int yn = clamp(idx.y + 1, 0, grid.height() - 1);
      int zn = clamp(idx.z + 1, 0, grid.depth() - 1);
      Vec3d pos = grid.indexToCoord(x, y, z);
      Real tx = clamp((p.x - pos.x) / grid.gridSpacing().x, 0.0, 1.0);
      Real ty = clamp((p.y - pos.y) / grid.gridSpacing().y, 0.0, 1.0);
      Real tz = clamp((p.z - pos.z) / grid.gridSpacing().z, 0.0, 1.0);
      assert(tx >= 0.0 && tx <= 1.0 && ty >= 0.0 && ty <= 1.0 && tz >= 0.0 &&
          tz <= 1.0);
      Real ret = tx * ty * tz * grid(x, y, z) +
                 tx * ty * (1 - tz) * grid(x, y, zn) +
                 tx * (1 - ty) * tz * grid(x, yn, z) +
                 tx * (1 - ty) * (1 - tz) * grid(x, yn, zn) +
                 (1 - tx) * ty * tz * grid(xn, y, z) +
                 (1 - tx) * ty * (1 - tz) * grid(xn, y, zn) +
                 (1 - tx) * (1 - ty) * tz * grid(xn, yn, z) +
                 (1 - tx) * (1 - ty) * (1 - tz) * grid(xn, yn, zn);
      return ret;
    }
  }
  [[nodiscard]] Vector<Real, Dim> gradient(const Vector<int, Dim>& idx) const {
    if constexpr (Dim == 2) {
    } else if constexpr (Dim == 3) {
      int xn = idx.x < width() - 1 ? idx.x + 1 : idx.x;
      int xp = idx.x > 0 ? idx.x - 1 : idx.x;
      int yn = idx.y < height() - 1 ? idx.y + 1 : idx.y;
      int yp = idx.y > 0 ? idx.y - 1 : idx.y;
      int zn = idx.z < depth() - 1 ? idx.z + 1 : idx.z;
      int zp = idx.z > 0 ? idx.z - 1 : idx.z;
      Real dx = idx.x > 0 && idx.x < width() - 1
                  ? 2 * grid.gridSpacing().x
                  : grid.gridSpacing().x;
      Real dy = idx.y > 0 && idx.y < height() - 1
                  ? 2 * grid.gridSpacing().y
                  : grid.gridSpacing().y;
      Real dz = idx.z > 0 && idx.z < depth() - 1
                  ? 2 * grid.gridSpacing().z
                  : grid.gridSpacing().z;
      Real gx = grid(xn, idx.y, idx.z) - grid(xp, idx.y, idx.z);
      Real gy = grid(idx.x, yn, idx.z) - grid(idx.x, yp, idx.z);
      Real gz = grid(idx.x, idx.y, zn) - grid(idx.x, idx.y, zp);
      return Vec3d(gx / dx, gy / dy, gz / dz);
    }
  }
  [[nodiscard]] Vector<Real, Dim> grad(const Vector<Real, Dim>& p) const {
    if constexpr (Dim == 2) {
      // use bilerp
    } else if constexpr (Dim == 3) {
      Vec3i idx = grid.coordToSampledCellIndex(p);
      // use trilerp to get the value
      int x = clamp(idx.x, 0, grid.width() - 1);
      int y = clamp(idx.y, 0, grid.height() - 1);
      int z = clamp(idx.z, 0, grid.depth() - 1);
      int xn = clamp(idx.x + 1, 0, grid.width() - 1);
      int yn = clamp(idx.y + 1, 0, grid.height() - 1);
      int zn = clamp(idx.z + 1, 0, grid.depth() - 1);
      Real tx = (p.x - grid.indexToCoord(x, 0, 0).x) / grid.gridSpacing().x;
      Real ty = (p.y - grid.indexToCoord(0, y, 0).y) / grid.gridSpacing().y;
      Real tz = (p.z - grid.indexToCoord(0, 0, z).z) / grid.gridSpacing().z;
      Vec3d g000 = gradient(Vec3i(x, y, z));
      Vec3d g001 = gradient(Vec3i(x, y, zn));
      Vec3d g010 = gradient(Vec3i(x, yn, z));
      Vec3d g011 = gradient(Vec3i(x, yn, zn));
      Vec3d g100 = gradient(Vec3i(xn, y, z));
      Vec3d g101 = gradient(Vec3i(xn, y, zn));
      Vec3d g110 = gradient(Vec3i(xn, yn, z));
      Vec3d g111 = gradient(Vec3i(xn, yn, zn));
      return tx * ty * tz * g000 +
             tx * ty * (1 - tz) * g001 +
             tx * (1 - ty) * tz * g010 +
             tx * (1 - ty) * (1 - tz) * g011 +
             (1 - tx) * ty * tz * g100 +
             (1 - tx) * ty * (1 - tz) * g101 +
             (1 - tx) * (1 - ty) * tz * g110 +
             (1 - tx) * (1 - ty) * (1 - tz) * g111;
    }
  }
  int width() const {
    return grid.width();
  }
  int height() const {
    return grid.height();
  }
  int depth() const {
    static_assert(Dim == 3);
    return grid.depth();
  }
  // for Rendering
  [[nodiscard]] std::vector<Vector<Real, Dim>> positionSamples() const {
    std::vector<Vector<Real, Dim>> position;
    if constexpr (Dim == 2) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          position.emplace_back(grid.indexToCoord(i, j));
    } else if constexpr (Dim == 3) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          for (int k = 0; k < grid.depth(); k++)
            position.emplace_back(grid.indexToCoord(i, j, k));
    }
    return position;
  }
  [[nodiscard]] std::vector<Vector<Real, 3>> fieldSamples() const {
    std::vector<Vector<Real, 3>> field;
    if constexpr (Dim == 2) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          field.emplace_back(grid(i, j), grid(i, j), grid(i, j));
    } else if constexpr (Dim == 3) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          for (int k = 0; k < grid.depth(); k++) {
            Real val = grid(i, j, k);
            field.emplace_back(val, val, val);
          }
    }
    return field;
  }
  void saveSDF(const std::string& filename) const {
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
      std::cerr << "Failed to open file " << filename << std::endl;
      return;
    }
    file << grid.origin().x << " " << grid.origin().y << " " << grid.origin().z << std::endl;
    file << grid.size().x << " " << grid.size().y << " " << grid.size().z << std::endl;
    file << grid.gridSpacing().x << " " << grid.gridSpacing().y << " " << grid.gridSpacing().z << std::endl;
    for (int i = 0; i < grid.width(); i++)
      for (int j = 0; j < grid.height(); j++)
        for (int k = 0; k < grid.depth(); k++)
          file << grid(i, j, k) << " ";
  }
  int sampleCount() const {
    return grid.width() * grid.height() * grid.depth();
  }
  const Vector<Real, Dim>& spacing() const {
    return grid.gridSpacing();
  }
  const Vector<Real, Dim>& origin() const {
    return grid.origin();
  }
  Real& operator()(int i, int j, int k) {
    return grid(i, j, k);
  }
  Real operator()(int i, int j, int k) const {
    return grid(i, j, k);
  }
  spatify::CellCentredGrid<Real, Real, Dim> grid;
};

template <typename T, int Dim>
struct ParticleSystemReconstructor : NonCopyable {
  static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
  virtual void reconstruct(std::span<Vector<T, Dim>> particles, Real radius,
                           SDF<3>& sdf, spatify::Array3D<char>& sdfValid) = 0;
  virtual ~ParticleSystemReconstructor() = default;
};

template <typename T, int Dim>
class NaiveReconstructor : ParticleSystemReconstructor<T, Dim> {
};

template <typename T>
class NaiveReconstructor<T, 3> final : public ParticleSystemReconstructor<T, 3> {
  public:
    NaiveReconstructor(int n, int w, int h, int d, const Vector<T, 3>& size)
      : ns(n, w, h, d, size) {
    }
    void reconstruct(std::span<Vector<T, 3>> particles, Real radius,
                     SDF<3>& sdf, spatify::Array3D<char>& sdfValid) override {
      ns.resetGrid(sdf.width(), sdf.height(), sdf.depth(),
                   sdf.grid.gridSpacing());
      ns.update(particles);
      sdf.grid.fill(1e9);
      sdfValid.fill(false);
      sdf.grid.parallelForEach([&](int i, int j, int k) {
        Vector<T, 3> p = sdf.grid.indexToCoord(i, j, k);
        ns.forNeighbours(p, particles, 2 * radius, [&](int idx) {
          Real dis = glm::distance(p, particles[idx]);
          sdf(i, j, k) = std::min(sdf(i, j, k), dis - radius);
          sdfValid(i, j, k) = true;
        });
      });
    }

  private:
    spatify::ParticleNeighbourSearcher<T, 3> ns;
};

void manifold2SDF(int exact_band, spatify::Array3D<int>& closest,
                  spatify::Array3D<int>& intersection_cnt,
                  const core::Mesh& mesh, SDF<3>* sdf);
std::optional<SDF<3>> loadSDF(const std::string& filename);
}
#endif //JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_