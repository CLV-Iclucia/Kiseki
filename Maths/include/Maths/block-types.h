#pragma once
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/geometric.hpp>
#include <array>
#include <cassert>
#include <Maths/block-vector.h>
#include <Maths/block-sparse-matrix.h>

namespace ksk::maths {

using Real = double;

template<int N>
struct LocalGrad {
  std::array<glm::dvec3, N> v{};

  LocalGrad() = default;
  LocalGrad(const std::array<glm::dvec3, N>& arr) : v(arr) {}
  LocalGrad(std::array<glm::dvec3, N>&& arr)       : v(std::move(arr)) {}

  glm::dvec3&       operator[](int i)       noexcept { return v[i]; }
  const glm::dvec3& operator[](int i) const noexcept { return v[i]; }

  auto begin()       noexcept { return v.begin(); }
  auto end()         noexcept { return v.end();   }
  auto begin() const noexcept { return v.begin(); }
  auto end()   const noexcept { return v.end();   }
  static constexpr std::size_t size() noexcept { return N; }
};

template<int N>
struct LocalHessian {
  std::array<std::array<glm::dmat3, N>, N> v{};

  LocalHessian() = default;
  LocalHessian(const std::array<std::array<glm::dmat3, N>, N>& arr) : v(arr) {}
  LocalHessian(std::array<std::array<glm::dmat3, N>, N>&& arr)      : v(std::move(arr)) {}

  std::array<glm::dmat3, N>&       operator[](int i)       noexcept { return v[i]; }
  const std::array<glm::dmat3, N>& operator[](int i) const noexcept { return v[i]; }
};

// ── 算符重载（类型在 ksk::maths 中，ADL 对所有 ksk:: 子命名空间均可见）───────────

template<int N>
inline LocalGrad<N> operator*(Real s, const LocalGrad<N>& g) {
  LocalGrad<N> r;
  for (int i = 0; i < N; i++) r[i] = s * g[i];
  return r;
}

template<int N>
inline LocalGrad<N> operator*(const LocalGrad<N>& g, Real s) { return s * g; }

template<int N>
inline LocalHessian<N> operator*(Real s, const LocalHessian<N>& h) {
  LocalHessian<N> r;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      r[i][j] = s * h[i][j];
  return r;
}

template<int N>
inline LocalHessian<N> operator*(const LocalHessian<N>& h, Real s) { return s * h; }

template<int N>
inline LocalHessian<N> operator+(const LocalHessian<N>& a, const LocalHessian<N>& b) {
  LocalHessian<N> r;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      r[i][j] = a[i][j] + b[i][j];
  return r;
}

/// 从 autogen 的 flat double[] 输出构造 LocalGrad
/// autogen 输出: [p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, ...]
template <int N>
LocalGrad<N> localGradFromFlat(const double* data) {
  LocalGrad<N> g;
  for (int i = 0; i < N; i++)
    g[i] = glm::dvec3(data[i*3], data[i*3+1], data[i*3+2]);
  return g;
}

/// 从 autogen 的 flat double[] 输出构造 LocalHessian
/// autogen 输出 row-major (3N × 3N) flat array:
///   data[row * D + col], where row = bi*3 + r, col = bj*3 + c
/// glm::dmat3 是 column-major: M[col][row]
template <int N>
LocalHessian<N> localHessianFromFlat(const double* data) {
  LocalHessian<N> H{};
  constexpr int D = N * 3;
  for (int bi = 0; bi < N; bi++) {
    for (int bj = 0; bj < N; bj++) {
      for (int c = 0; c < 3; c++) {      // glm col
        for (int r = 0; r < 3; r++) {    // glm row
          H[bi][bj][c][r] = data[(bi*3 + r) * D + (bj*3 + c)];
        }
      }
    }
  }
  return H;
}

/// 将 LocalGrad 累加到 BlockVector<3>
template <int N>
void assembleLocalGrad(BlockVector<3>& out,
                       const std::array<int, N>& globalIdx,
                       const LocalGrad<N>& grad,
                       Real scale = 1.0) {
  for (int i = 0; i < N; i++)
    out[globalIdx[i]] += grad[i] * scale;
}

/// 将 LocalHessian 组装到 BlockSparseMatrix<3>
/// H_barrier[i][j] = kappa * (bHess * outerProduct(grad[i], grad[j]) + bGrad * hess[i][j])
template <int N>
void assembleLocalHessian(BlockSparseMatrix<3>& out,
                          const std::array<int, N>& globalIdx,
                          const LocalHessian<N>& hess,
                          const LocalGrad<N>& grad,
                          Real bGrad,
                          Real bHess,
                          Real kappa = 1.0) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // outerProduct: grad[i] ⊗ grad[j] → dmat3
      glm::dmat3 outerGG = glm::outerProduct(grad[i], grad[j]);
      glm::dmat3 block = kappa * (bHess * outerGG + bGrad * hess[i][j]);
      out.addBlock(globalIdx[i], globalIdx[j], block);
    }
  }
}

/// 将 LocalHessian 组装到 BlockSparseMatrix<3> (无 barrier 的简化版本)
template <int N>
void assembleLocalHessian(BlockSparseMatrix<3>& out,
                          const std::array<int, N>& globalIdx,
                          const LocalHessian<N>& hess,
                          Real scale = 1.0) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      out.addBlock(globalIdx[i], globalIdx[j], hess[i][j] * scale);
    }
  }
}

/// 从 LocalGrad 构造 outer product 矩阵: result[i][j] = outer(grad[i], grad[j])
template <int N>
LocalHessian<N> outerProductMatrix(const LocalGrad<N>& grad) {
  LocalHessian<N> result{};
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] = glm::outerProduct(grad[i], grad[j]);
    }
  }
  return result;
}

} // namespace ksk::maths
