//
// Created by creeper on 7/14/24.
//

#include <fem/ipc/distances.h>
#include <Maths/block-types.h>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <stdexcept>

namespace sim::fem::ipc {

using maths::LocalGrad;
using maths::LocalHessian;

// =========================================================================
// distanceSqrPointTriangle — 标量返回值，glm 接口
// =========================================================================

Real distanceSqrPointTriangle(const glm::dvec3 &p,
                              const glm::dvec3 &a,
                              const glm::dvec3 &b,
                              const glm::dvec3 &c) {
  // Logic shared with the GPU via <fem/shared/ipc-distance.h>.
  return shared::shDistanceSqrPointTriangle(p, a, b, c);
}

// =========================================================================
// localDistancePointTriangleGradient — 返回 LocalGrad<4>
// =========================================================================

LocalGrad<4> localDistancePointTriangleGradient(const glm::dvec3 &p,
                                                    const glm::dvec3 &a,
                                                    const glm::dvec3 &b,
                                                    const glm::dvec3 &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  LocalGrad<4> grad{};  // zero-initialized
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto g = localDistanceSqrPointPointGradient(p, a);
      grad[0] = g[0];  // p
      grad[1] = g[1];  // a
      return grad;
    }
    case PointTriangleDistanceType::P_B: {
      auto g = localDistanceSqrPointPointGradient(p, b);
      grad[0] = g[0];  // p
      grad[2] = g[1];  // b
      return grad;
    }
    case PointTriangleDistanceType::P_C: {
      auto g = localDistanceSqrPointPointGradient(p, c);
      grad[0] = g[0];  // p
      grad[3] = g[1];  // c
      return grad;
    }
    case PointTriangleDistanceType::P_AB: {
      auto g = localDistanceSqrPointLineGradient(p, a, b);
      // g: [p, a, b] → grad: [p@0, a@1, b@2, 0@3]
      grad[0] = g[0];
      grad[1] = g[1];
      grad[2] = g[2];
      return grad;
    }
    case PointTriangleDistanceType::P_BC: {
      auto g = localDistanceSqrPointLineGradient(p, b, c);
      // g: [p, b, c] → grad: [p@0, 0@1, b@2, c@3]
      grad[0] = g[0];
      grad[2] = g[1];
      grad[3] = g[2];
      return grad;
    }
    case PointTriangleDistanceType::P_CA: {
      auto g = localDistanceSqrPointLineGradient(p, c, a);
      // g: [p, c, a] → grad: [p@0, a@1, 0@2, c@3]
      grad[0] = g[0];
      grad[1] = g[2];
      grad[3] = g[1];
      return grad;
    }
    case PointTriangleDistanceType::P_ABC:
      return localDistanceSqrPointPlaneGradient(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}

// =========================================================================
// localDistancePointTriangleHessian — 返回 LocalHessian<4>
// =========================================================================

LocalHessian<4> localDistancePointTriangleHessian(const glm::dvec3 &p,
                                                       const glm::dvec3 &a,
                                                       const glm::dvec3 &b,
                                                       const glm::dvec3 &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  LocalHessian<4> H{};  // zero-initialized
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto h = localDistanceSqrPointPointHessian(p, a);
      // h: [p, a] × [p, a]
      H[0][0] = h[0][0];
      H[0][1] = h[0][1];
      H[1][0] = h[1][0];
      H[1][1] = h[1][1];
      return H;
    }
    case PointTriangleDistanceType::P_B: {
      auto h = localDistanceSqrPointPointHessian(p, b);
      H[0][0] = h[0][0];  // p-p
      H[0][2] = h[0][1];  // p-b
      H[2][0] = h[1][0];  // b-p
      H[2][2] = h[1][1];  // b-b
      return H;
    }
    case PointTriangleDistanceType::P_C: {
      auto h = localDistanceSqrPointPointHessian(p, c);
      H[0][0] = h[0][0];  // p-p
      H[0][3] = h[0][1];  // p-c
      H[3][0] = h[1][0];  // c-p
      H[3][3] = h[1][1];  // c-c
      return H;
    }
    case PointTriangleDistanceType::P_AB: {
      auto h = localDistanceSqrPointLineHessian(p, a, b);
      // h: [p, a, b] × [p, a, b]
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          H[i][j] = h[i][j];
      return H;
    }
    case PointTriangleDistanceType::P_BC: {
      auto h = localDistanceSqrPointLineHessian(p, b, c);
      // h: [p, b, c] → H: [p@0, 0@1, b@2, c@3]
      for (int r = 0; r < 3; r++) {
        H[0][0][r] = h[0][0][r];
        H[0][2][r] = h[0][1][r];
        H[0][3][r] = h[0][2][r];
        H[2][0][r] = h[1][0][r];
        H[2][2][r] = h[1][1][r];
        H[2][3][r] = h[1][2][r];
        H[3][0][r] = h[2][0][r];
        H[3][2][r] = h[2][1][r];
        H[3][3][r] = h[2][2][r];
      }
      return H;
    }
    case PointTriangleDistanceType::P_CA: {
      auto h = localDistanceSqrPointLineHessian(p, c, a);
      // h: [p, c, a] → H: [p@0, a@1, 0@2, c@3]
      for (int r = 0; r < 3; r++) {
        H[0][0][r] = h[0][0][r];
        H[0][1][r] = h[0][2][r];
        H[0][3][r] = h[0][1][r];
        H[1][0][r] = h[2][0][r];
        H[1][1][r] = h[2][2][r];
        H[1][3][r] = h[2][1][r];
        H[3][0][r] = h[1][0][r];
        H[3][1][r] = h[1][2][r];
        H[3][3][r] = h[1][1][r];
      }
      return H;
    }
    case PointTriangleDistanceType::P_ABC:
      return localDistanceSqrPointPlaneHessian(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}

// =========================================================================
// distanceSqrEdgeEdge — 标量返回值，glm 接口
// =========================================================================

Real distanceSqrEdgeEdge(const glm::dvec3 &ea0,
                         const glm::dvec3 &ea1,
                         const glm::dvec3 &eb0,
                         const glm::dvec3 &eb1) {
  // Logic shared with the GPU via <fem/shared/ipc-distance.h>.
  return shared::shDistanceSqrEdgeEdge(ea0, ea1, eb0, eb1);
}

// =========================================================================
// localDistanceSqrEdgeEdgeGradient — 返回 LocalGrad<4>
// =========================================================================

LocalGrad<4> localDistanceSqrEdgeEdgeGradient(const glm::dvec3 &a,
                                                  const glm::dvec3 &b,
                                                  const glm::dvec3 &c,
                                                  const glm::dvec3 &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  LocalGrad<4> grad{};
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      auto g = localDistanceSqrPointPointGradient(a, c);
      grad[0] = g[0];  // a
      grad[2] = g[1];  // c
      return grad;
    }
    case EdgeEdgeDistanceType::A_D: {
      auto g = localDistanceSqrPointPointGradient(a, d);
      grad[0] = g[0];  // a
      grad[3] = g[1];  // d
      return grad;
    }
    case EdgeEdgeDistanceType::B_C: {
      auto g = localDistanceSqrPointPointGradient(b, c);
      grad[1] = g[0];  // b
      grad[2] = g[1];  // c
      return grad;
    }
    case EdgeEdgeDistanceType::B_D: {
      auto g = localDistanceSqrPointPointGradient(b, d);
      grad[1] = g[0];  // b
      grad[3] = g[1];  // d
      return grad;
    }
    case EdgeEdgeDistanceType::A_CD: {
      auto g = localDistanceSqrPointLineGradient(a, c, d);
      // g: [a, c, d]
      grad[0] = g[0];
      grad[2] = g[1];
      grad[3] = g[2];
      return grad;
    }
    case EdgeEdgeDistanceType::B_CD: {
      auto g = localDistanceSqrPointLineGradient(b, c, d);
      grad[1] = g[0];
      grad[2] = g[1];
      grad[3] = g[2];
      return grad;
    }
    case EdgeEdgeDistanceType::AB_C: {
      auto g = localDistanceSqrPointLineGradient(c, a, b);
      // g: [c, a, b]
      grad[2] = g[0];
      grad[0] = g[1];
      grad[1] = g[2];
      return grad;
    }
    case EdgeEdgeDistanceType::AB_D: {
      auto g = localDistanceSqrPointLineGradient(d, a, b);
      // g: [d, a, b]
      grad[3] = g[0];
      grad[0] = g[1];
      grad[1] = g[2];
      return grad;
    }
    case EdgeEdgeDistanceType::AB_CD:
      return localDistanceSqrLineLineGradient(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}

// =========================================================================
// localDistanceSqrEdgeEdgeHessian — 返回 LocalHessian<4>
// =========================================================================

LocalHessian<4> localDistanceSqrEdgeEdgeHessian(const glm::dvec3 &a,
                                                     const glm::dvec3 &b,
                                                     const glm::dvec3 &c,
                                                     const glm::dvec3 &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  LocalHessian<4> H{};
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      auto h = localDistanceSqrPointPointHessian(a, c);
      H[0][0] = h[0][0];  // a-a
      H[0][2] = h[0][1];  // a-c
      H[2][0] = h[1][0];  // c-a
      H[2][2] = h[1][1];  // c-c
      return H;
    }
    case EdgeEdgeDistanceType::A_D: {
      auto h = localDistanceSqrPointPointHessian(a, d);
      H[0][0] = h[0][0];  // a-a
      H[0][3] = h[0][1];  // a-d
      H[3][0] = h[1][0];  // d-a
      H[3][3] = h[1][1];  // d-d
      return H;
    }
    case EdgeEdgeDistanceType::B_C: {
      auto h = localDistanceSqrPointPointHessian(b, c);
      H[1][1] = h[0][0];  // b-b
      H[1][2] = h[0][1];  // b-c
      H[2][1] = h[1][0];  // c-b
      H[2][2] = h[1][1];  // c-c
      return H;
    }
    case EdgeEdgeDistanceType::B_D: {
      auto h = localDistanceSqrPointPointHessian(b, d);
      H[1][1] = h[0][0];  // b-b
      H[1][3] = h[0][1];  // b-d
      H[3][1] = h[1][0];  // d-b
      H[3][3] = h[1][1];  // d-d
      return H;
    }
    case EdgeEdgeDistanceType::A_CD: {
      auto h = localDistanceSqrPointLineHessian(a, c, d);
      // h: [a, c, d] × [a, c, d]
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          H[i][j] = h[i][j];
      // H[0][2], H[0][3], H[2][0], H[2][3], H[3][0], H[3][2] already set
      // Need to map: h[0]=a, h[1]=c, h[2]=d → H[0]=a, H[2]=c, H[3]=d
      for (int r = 0; r < 3; r++) {
        H[0][0][r] = h[0][0][r];
        H[0][2][r] = h[0][1][r];
        H[0][3][r] = h[0][2][r];
        H[2][0][r] = h[1][0][r];
        H[2][2][r] = h[1][1][r];
        H[2][3][r] = h[1][2][r];
        H[3][0][r] = h[2][0][r];
        H[3][2][r] = h[2][1][r];
        H[3][3][r] = h[2][2][r];
      }
      return H;
    }
    case EdgeEdgeDistanceType::B_CD: {
      auto h = localDistanceSqrPointLineHessian(b, c, d);
      for (int r = 0; r < 3; r++) {
        H[1][1][r] = h[0][0][r];
        H[1][2][r] = h[0][1][r];
        H[1][3][r] = h[0][2][r];
        H[2][1][r] = h[1][0][r];
        H[2][2][r] = h[1][1][r];
        H[2][3][r] = h[1][2][r];
        H[3][1][r] = h[2][0][r];
        H[3][2][r] = h[2][1][r];
        H[3][3][r] = h[2][2][r];
      }
      return H;
    }
    case EdgeEdgeDistanceType::AB_C: {
      auto h = localDistanceSqrPointLineHessian(c, a, b);
      // h: [c, a, b] → H: [a@0, b@1, c@2, 0@3]
      for (int r = 0; r < 3; r++) {
        H[2][2][r] = h[0][0][r];
        H[2][0][r] = h[0][1][r];
        H[2][1][r] = h[0][2][r];
        H[0][2][r] = h[1][0][r];
        H[0][0][r] = h[1][1][r];
        H[0][1][r] = h[1][2][r];
        H[1][2][r] = h[2][0][r];
        H[1][0][r] = h[2][1][r];
        H[1][1][r] = h[2][2][r];
      }
      return H;
    }
    case EdgeEdgeDistanceType::AB_D: {
      auto h = localDistanceSqrPointLineHessian(d, a, b);
      // h: [d, a, b] → H: [a@0, b@1, 0@2, d@3]
      for (int r = 0; r < 3; r++) {
        H[3][3][r] = h[0][0][r];
        H[3][0][r] = h[0][1][r];
        H[3][1][r] = h[0][2][r];
        H[0][3][r] = h[1][0][r];
        H[0][0][r] = h[1][1][r];
        H[0][1][r] = h[1][2][r];
        H[1][3][r] = h[2][0][r];
        H[1][0][r] = h[2][1][r];
        H[1][1][r] = h[2][2][r];
      }
      return H;
    }
    case EdgeEdgeDistanceType::AB_CD:
      return localDistanceSqrLineLineHessian(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}

} // namespace sim::fem::ipc
