//
// Created by CreeperIclucia-Vader on 25-5-16.
//

#pragma once

#include <span>
#include <fem/simplex.h>
#include <Maths/block-vector.h>

namespace ksk::fem {

struct PointTrajectory {
  const maths::BlockVector<3> &x;
  const maths::BlockVector<3> &p;
  Real toi = 1.0;
  int global_idx{};
  [[nodiscard]] BBox<Real, 3> bbox() const {
    auto v = x[global_idx];
    auto u = p[global_idx] * toi;
    return BBox<Real, 3>({v.x, v.y, v.z}).expand({v.x + u.x, v.y + u.y, v.z + u.z});
  }
};

struct TriangleTrajectory {
  const maths::BlockVector<3> &x;
  const maths::BlockVector<3> &p;
  Real toi = 1.0;
  Triangle triangle{};
  
  [[nodiscard]] BBox<Real, 3> bbox() const {
    BBox<Real, 3> box;
    int verts[3] = {triangle.x, triangle.y, triangle.z};
    for (int i = 0; i < 3; i++) {
      const auto& pos = x[verts[i]];
      auto endPos = pos + p[verts[i]] * toi;
      box.expand({pos.x, pos.y, pos.z});
      box.expand({endPos.x, endPos.y, endPos.z});
    }
    return box;
  }
};

}