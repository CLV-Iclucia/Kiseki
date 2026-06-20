#include <Core/mesh.h>
#include <FluidSim/cpu/util.h>
#include <FluidSim/kernel.h>
#include <FluidSim/cpu/sdf.h>
#include <iostream>

namespace fluid {
Vec2d sampleVelocity(const Vec2d& p,
                     const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                     const FaceCentredGrid<Real, Real, 2, 1>* v_grid) {
  Vec2i u_idx = u_grid->nearest(p);
  Vec2i v_idx = v_grid->nearest(p);
  Real h = u_grid->gridSpacing().x;
  Real u = 0.0, v = 0.0;
  Real w_u = 0.0, w_v = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int k = -1; k <= 1; k++) {
      if (u_idx.x + j >= 0 && u_idx.x + j < u_grid->width() &&
          u_idx.y + k >= 0 && u_idx.y + k < u_grid->height()) {
        Real un = u_grid->at(u_idx + Vec2i(j, k));
        Vec2d pn = u_grid->indexToCoord(u_idx + Vec2i(j, k));
        Real w = cubicWeight2D(h, p - pn);
        u += un * w;
        w_u += w;
      }
      if (v_idx.x + j >= 0 && v_idx.x + j < v_grid->width() &&
          v_idx.y + k >= 0 && v_idx.y + k < v_grid->height()) {
        Real vn = v_grid->at(v_idx + Vec2i(j, k));
        Vec2d pn = v_grid->indexToCoord(v_idx + Vec2i(j, k));
        Real w = cubicWeight2D(h, p - pn);
        v += vn * w;
        w_v += w;
      }
    }
  }
  return Vec2d(w_u > 0.0 ? u / w_u : 0.0, w_v > 0.0 ? v / w_v : 0.0);
}

Vec3d sampleVelocity(const Vec3d& p,
                     const FaceCentredGrid<Real, Real, 3, 0>& ug,
                     const FaceCentredGrid<Real, Real, 3, 1>& vg,
                     const FaceCentredGrid<Real, Real, 3, 2>& wg) {
  Vec3i u_idx = ug.nearest(p);
  Vec3i v_idx = vg.nearest(p);
  Vec3i w_idx = wg.nearest(p);
  Real h = ug.gridSpacing().x;
  Real u = 0.0, v = 0.0, w = 0.0;
  Real w_u = 0.0, w_v = 0.0, w_w = 0.0;
  for (int i = -1; i <= 2; i++) {
    for (int j = -1; j <= 2; j++) {
      for (int k = -1; k <= 2; k++) {
        if (u_idx.x + i >= 0 && u_idx.x + i < ug.width() &&
            u_idx.y + j >= 0 && u_idx.y + j < ug.height() &&
            u_idx.z + k >= 0 && u_idx.z + k < ug.depth()) {
          Real un = ug(u_idx + Vec3i(i, j, k));
          Vec3d pn = ug.indexToCoord(u_idx + Vec3i(i, j, k));
          Real uw = cubicWeight3D(h, p - pn);
          u += un * uw;
          w_u += uw;
        }
        if (v_idx.x + i >= 0 && v_idx.x + i < vg.width() &&
            v_idx.y + j >= 0 && v_idx.y + j < vg.height() &&
            v_idx.z + k >= 0 && v_idx.z + k < vg.depth()) {
          Real vn = vg(v_idx + Vec3i(i, j, k));
          Vec3d pn = vg.indexToCoord(v_idx + Vec3i(i, j, k));
          Real vw = cubicWeight3D(h, p - pn);
          v += vn * vw;
          w_v += vw;
        }
        if (w_idx.x + i >= 0 && w_idx.x + i < wg.width() &&
            w_idx.y + j >= 0 && w_idx.y + j < wg.height() &&
            w_idx.z + k >= 0 && w_idx.z + k < wg.depth()) {
          Real wn = wg.at(w_idx + Vec3i(i, j, k));
          Vec3d pn = wg.indexToCoord(w_idx + Vec3i(i, j, k));
          Real ww = cubicWeight3D(h, p - pn);
          w += wn * ww;
          w_w += ww;
        }
      }
    }
  }
  return Vec3d(w_u > 0.0 ? u / w_u : 0.0, w_v > 0.0 ? v / w_v : 0.0,
               w_w > 0.0 ? w / w_w : 0.0);
}

static Real distancePointTriangle(const Vec3d& p, const Vec3d& a,
                                  const Vec3d& b,
                                  const Vec3d& c) {
  Vec3d ab = b - a;
  Vec3d ac = c - a;
  Vec3d ap = p - a;
  Real d1 = glm::dot(ab, ap);
  Real d2 = glm::dot(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return glm::length(ap);
  Vec3d bp = p - b;
  Real d3 = glm::dot(ab, bp);
  Real d4 = glm::dot(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return glm::length(bp);
  Real vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    Real v = d1 / (d1 - d3);
    return glm::length(ap + v * ab);
  }
  Vec3d cp = p - c;
  Real d5 = glm::dot(ab, cp);
  Real d6 = glm::dot(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return glm::length(cp);
  Real vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    Real w = d2 / (d2 - d6);
    return glm::length(ap + w * ac);
  }
  Real va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    Real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return glm::length(bp + w * (cp - bp));
  }
  Real denom = 1.0 / (va + vb + vc);
  Real v = vb * denom;
  Real w = vc * denom;
  return glm::length(ap + ab * v + ac * w);
}

static Real cross(const Vec2d& a, const Vec2d& b) {
  return a.x * b.y - b.x * a.y;
}

static bool insideTraingle2D(const Vec2d& p, const Vec2d& a, const Vec2d& b,
                             const Vec2d& c) {
  Vec2d ap = p - a;
  Vec2d ab = b - a;
  Vec2d bp = p - b;
  Vec2d bc = c - b;
  Vec2d cp = p - c;
  Vec2d ca = a - c;
  int cnt = (cross(ap, ab) >= 0.0) + (cross(bp, bc) >= 0.0) + (
              cross(cp, ca) >= 0.0);
  return cnt == 3 || cnt == 0;
}

static void solveEquations(Real a1, Real b1, Real c1, Real a2, Real b2, Real c2,
                           Real* u, Real* v) {
  Real determinant = a1 * b2 - a2 * b1;
  *u = (c1 * b2 - c2 * b1) / determinant;
  *v = (a1 * c2 - a2 * c1) / determinant;
}

static void calcBiocentricCoord(const Vec2d& p, const Vec3d& a, const Vec3d& b,
                                const Vec3d& c, Real* u, Real* v, Real* w) {
  solveEquations(a.x - c.x, b.x - c.x, p.x - c.x, a.y - c.y, b.y - c.y,
                 p.y - c.y, u, v);
  *w = 1.0 - (*u) - (*v);
}

static void checkNeighbour(SDF<3>* sdf, const core::Mesh& mesh,
                           spatify::Array3D<int>& closest,
                           int dx, int dy, int dz, int x, int y, int z) {
  if (x + dx < 0 || x + dx >= sdf->width() || y + dy < 0 || y + dy >= sdf->
      height() || z + dz < 0 || z + dz >= sdf->depth())
    return;
  if (closest(x + dx, y + dy, z + dz) == -1)
    return;
  int tri_idx = closest(x + dx, y + dy, z + dz);
  Vec3d p = sdf->grid.indexToCoord(x, y, z);
  Real d = sdf->grid(x, y, z);
  Vec3d a = mesh.vertices[mesh.indices[3 * tri_idx]];
  Vec3d b = mesh.vertices[mesh.indices[3 * tri_idx + 1]];
  Vec3d c = mesh.vertices[mesh.indices[3 * tri_idx + 2]];
  Real d1 = distancePointTriangle(p, a, b, c);
  if (abs(d1) < abs(d)) {
    sdf->grid(x, y, z) = d1;
    closest(x, y, z) = closest(x + dx, y + dy, z + dz);
  }
}

static void sweep(SDF<3>* sdf, const core::Mesh& mesh,
                  spatify::Array3D<int>& closest, int dx, int dy, int dz) {
  int x_start = dx < 0 ? 0 : sdf->width() - 1;
  int x_end = dx < 0 ? sdf->width() : -1;
  int y_start = dy < 0 ? 0 : sdf->height() - 1;
  int y_end = dy < 0 ? sdf->height() : -1;
  int z_start = dz < 0 ? 0 : sdf->depth() - 1;
  int z_end = dz < 0 ? sdf->depth() : -1;
  for (int x = x_start; x != x_end; x -= dx) {
    for (int y = y_start; y != y_end; y -= dy) {
      for (int z = z_start; z != z_end; z -= dz) {
        checkNeighbour(sdf, mesh, closest, 0, dy, dz, x, y, z);
        checkNeighbour(sdf, mesh, closest, dx, 0, dz, x, y, z);
        checkNeighbour(sdf, mesh, closest, dx, dy, 0, x, y, z);
        checkNeighbour(sdf, mesh, closest, 0, 0, dz, x, y, z);
        checkNeighbour(sdf, mesh, closest, 0, dy, 0, x, y, z);
        checkNeighbour(sdf, mesh, closest, dx, 0, 0, x, y, z);
        checkNeighbour(sdf, mesh, closest, dx, dy, dz, x, y, z);
      }
    }
  }
}

void manifold2SDF(int exact_band, spatify::Array3D<int>& closest,
                  spatify::Array3D<int>& intersection_cnt,
                  const core::Mesh& mesh, SDF<3>* sdf) {
  int width = sdf->width();
  int height = sdf->height();
  int depth = sdf->depth();
  closest.fill(-1);
  intersection_cnt.fill(0);
  sdf->grid.fill(1e9);
  Real h = sdf->spacing().x;
  Real ox = sdf->origin().x;
  Real oy = sdf->origin().y;
  Real oz = sdf->origin().z;
  for (int i = 0; i < mesh.triangleCount; i++) {
    Vec3d a = mesh.vertices[mesh.indices[3 * i]], b = mesh.vertices[
      mesh.indices[3 * i + 1]], c =
        mesh.vertices[mesh.indices[3 * i + 2]];
    int lo_x, hi_x, lo_y, hi_y, lo_z, hi_z;
    lo_x = clamp(
        static_cast<int>(std::floor(
            std::min(std::min(a.x - ox, b.x - ox), c.x - ox) / h - 0.5)) -
        exact_band, 0, width - 1);
    hi_x = clamp(
        static_cast<int>(std::ceil(
            std::max(std::max(a.x - ox, b.x - ox), c.x - ox) / h - 0.5)) +
        exact_band, 0, width - 1);
    lo_y = clamp(
        static_cast<int>(std::floor(
            std::min(std::min(a.y - oy, b.y - oy), c.y - oy) / h - 0.5)) -
        exact_band, 0, height - 1);
    hi_y = clamp(
        static_cast<int>(std::ceil(
            std::max(std::max(a.y - oy, b.y - oy), c.y - oy) / h - 0.5)) +
        exact_band, 0, height - 1);
    lo_z = clamp(
        static_cast<int>(std::floor(
            std::min(std::min(a.z - oz, b.z - oz), c.z - oz) / h - 0.5)) -
        exact_band, 0, depth - 1);
    hi_z = clamp(
        static_cast<int>(std::ceil(
            std::max(std::max(a.z - oz, b.z - oz), c.z - oz) / h - 0.5)) +
        exact_band, 0, depth - 1);
    for (int x = lo_x; x <= hi_x; x++) {
      for (int y = lo_y; y <= hi_y; y++) {
        for (int z = lo_z; z <= hi_z; z++) {
          Vec3d p = sdf->grid.indexToCoord(x, y, z);
          Real d = distancePointTriangle(p, a, b, c);
          Real d1 = sdf->grid(x, y, z);
          if (d < d1) {
            sdf->grid(x, y, z) = d;
            closest(x, y, z) = i;
          }
        }
      }
    }
    lo_x = clamp(
        static_cast<int>(std::floor(
            std::min(std::min(a.x - ox, b.x - ox), c.x - ox) / h - 0.5)), 0,
        width - 1);
    hi_x = clamp(
        static_cast<int>(std::ceil(
            std::max(std::max(a.x - ox, b.x - ox), c.x - ox) / h - 0.5)), 0,
        width - 1);
    lo_y = clamp(
        static_cast<int>(std::floor(
            std::min(std::min(a.y - oy, b.y - oy), c.y - oy) / h - 0.5)), 0,
        height - 1);
    hi_y = clamp(
        static_cast<int>(std::ceil(
            std::max(std::max(a.y - oy, b.y - oy), c.y - oy) / h - 0.5)), 0,
        height - 1);
    // project the triangle onto the plane x-y
    Vec2d proj_a(a.x, a.y), proj_b(b.x, b.y), proj_c(c.x, c.y);
    // that is the bounding box of the triangle
    for (int j = lo_x; j <= hi_x; j++) {
      for (int k = lo_y; k <= hi_y; k++) {
        Vec3d orig = sdf->grid.indexToCoord(j, k, -1);
        Vec2d proj_p(orig.x, orig.y);
        if (!insideTraingle2D(proj_p, proj_a, proj_b, proj_c)) continue;
        Real u = -1.0, v = -1.0, w = -1.0;
        calcBiocentricCoord(proj_p, a, b, c, &u, &v, &w);
        assert(
            u >= 0.0 && v >= 0.0 && w >= 0.0 && u <= 1.0 && v <= 1.0 && w <=
            1.0);
        Real inter_z = a.z * u + b.z * v + c.z * w;
        int l = static_cast<int>(std::ceil((inter_z - oz) / h - 0.5));
        assert(
            (l + 0.5) * h + oz >= inter_z && (l - 0.5) * h + oz < inter_z);
        if (l < 0) intersection_cnt(j, k, 0)++;
        else if (l < depth) intersection_cnt(j, k, l)++;
      }
    }
  }
  // do sweep
  for (int round = 0; round < 2; round++) {
    sweep(sdf, mesh, closest, 1, 1, 1);
    sweep(sdf, mesh, closest, 1, 1, -1);
    sweep(sdf, mesh, closest, 1, -1, 1);
    sweep(sdf, mesh, closest, 1, -1, -1);
    sweep(sdf, mesh, closest, -1, 1, 1);
    sweep(sdf, mesh, closest, -1, 1, -1);
    sweep(sdf, mesh, closest, -1, -1, 1);
    sweep(sdf, mesh, closest, -1, -1, -1);
  }
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int tot_cnt = 0;
      for (int k = 0; k < depth; k++) {
        tot_cnt += intersection_cnt(i, j, k);
        if (tot_cnt & 1)
          sdf->grid(i, j, k) = -sdf->grid(i, j, k);
      }
    }
  }
}

static void cycle(Real& a0, Real& a1, Real& a2, Real& a3) {
  Real tmp = a0;
  a0 = a3;
  a3 = a2;
  a2 = a1;
  a1 = tmp;
}

// lu---------ru
//  |         |
//  |         |
//  |         |
//  |         |
// lb---------rb
Real fractionInside(Real lu, Real ru, Real rd, Real ld) {
  int n_negs = (lu <= 0.0) + (ru <= 0.0) + (rd <= 0.0) + (ld <= 0.0);
  if (n_negs == 0)
    return 0.0;
  if (n_negs == 4)
    return 1.0;
  while (lu > 0.0)
    cycle(lu, ru, rd, ld);
  // now lu must be negative
  if (n_negs == 1) {
    Real fracu = ru / (ru - lu);
    Real fracl = ld / (ld - lu);
    return fracu * fracl * 0.5;
  }
  if (n_negs == 2) {
    if (ru <= 0.0) {
      Real fracl = ld / (ld - lu);
      Real fracr = rd / (rd - ru);
      return 0.5 * (fracl + fracr);
    }
    if (ld <= 0.0) {
      Real fracu = ru / (ru - lu);
      Real fracd = rd / (rd - ld);
      return 0.5 * (fracu + fracd);
    }
    if (rd <= 0.0) {
      Real fracu = ru / (ru - lu);
      Real fracd = ld / (ld - rd);
      Real fracl = ld / (ld - lu);
      Real fracr = ru / (ru - rd);
      if (lu + ru + rd + ld <= 0.0)
        return 1.0 - 0.5 * ((1.0 - fracu) * (1.0 - fracl) + (1.0 - fracd) * (
                              1.0 - fracr));
      return 0.5 * (fracu * fracl + fracd * fracr);
    }
    std::cerr << "Error: numerical error" << std::endl;
  }
  if (n_negs == 3) {
    while (lu <= 0.0)
      cycle(lu, ru, rd, ld);
    // now lu is the only positive value
    Real fracu = ru / (ru - lu);
    Real fracl = ld / (ld - lu);
    return 1.0 - 0.5 * fracu * fracl;
  }
}

Real dotProduct(const spatify::Array3D<Real>& a,
                const spatify::Array3D<Real>& b,
                const spatify::Array3D<uint8_t>& active) {
  Real sum = 0.0;
  a.forEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    assert(!std::isnan(a(i, j, k)));
    assert(!std::isnan(b(i, j, k)));
    sum += a(i, j, k) * b(i, j, k);
  });
  assert(!std::isnan(sum));
  return sum;
}

void saxpy(spatify::Array3D<Real>& a, const spatify::Array3D<Real>& b, Real x,
           const spatify::Array3D<uint8_t>& active) {
  assert(!std::isnan(x));
  a.parallelForEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    a(i, j, k) += x * b(i, j, k);
    assert(!std::isnan(a(i, j, k)));
  });
}

void scaleAndAdd(spatify::Array3D<Real>& c, const spatify::Array3D<Real>& a,
                 Real x, const spatify::Array3D<uint8_t>& active) {
  c.parallelForEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    c(i, j, k) = a(i, j, k) + x * c(i, j, k);
    assert(!std::isnan(c(i, j, k)));
  });
}
std::optional<SDF<3>> loadSDF(const std::string& filename) {
  std::fstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file " << filename << std::endl;
    return std::nullopt;
  }
  Vector<int, 3> size;
  Vector<Real, 3> origin, spacing;
  file >> origin.x >> origin.y >> origin.z;
  file >> size.x >> size.y >> size.z;
  file >> spacing.x >> spacing.y >> spacing.z;
  auto sdf = std::make_optional<SDF<3>>(size, Vec3d(spacing.x * size.x, spacing.y * size.y, spacing.z * size.z), origin);
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++)
      for (int k = 0; k < size.z; k++)
        file >> (*sdf)(i, j, k);
  return std::move(sdf);
}
} // namespace fluid