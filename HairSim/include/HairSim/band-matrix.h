#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_BANDED_MATRIX_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_BANDED_MATRIX_H_

#include <HairSim/hair-sim.h>
#include <mkl.h>
#include <vector>
#include <cassert>

namespace hairsim {
template <typename T, int BandWidth>
class BandSquareMatrix {
  public:
    explicit
    BandSquareMatrix(int n_) : m_data((3 * BandWidth + 1) * n_), n(n_) {
    }
    T& operator()(int i, int j) {
      assert(j - i <= BandWidth && i - j <= BandWidth);
      assert(i >= 0 && i < n && j >= 0 && j < n);
      return m_data[(i - j + 2 * BandWidth) * n + j];
    }
    T operator()(int i, int j) const {
      assert(j - i <= BandWidth && i - j <= BandWidth);
      assert(i >= 0 && i < n && j >= 0 && j < n);
      return m_data[(i - j + 2 * BandWidth) * n + j];
    }
    const T* data() const { return m_data.data(); }
    T* data() { return m_data.data(); }
    [[nodiscard]] int order() const { return n; }
    [[nodiscard]] int rows() const { return n; }
    [[nodiscard]] int cols() const { return n; }

  private:
    std::vector<T> m_data;
    int n;
};

template <typename T, int BandWidth>
class BandLUSolver {
  public:
    void solve(BandSquareMatrix<T, BandWidth>& A,
               const VecXd& rhs, VecXd& x) {
      assert(rhs.rows() == A.order());
      x = rhs;
      if (A.order() > ipiv.size())
        ipiv.resize(A.order());
      result = LAPACKE_dgbsv(
          LAPACK_ROW_MAJOR, A.order(), BandWidth,
          BandWidth, 1, A.data(), A.order(),
          ipiv.data(), x.data(), 1);
    }
    [[nodiscard]] int info() const { return result; }
    [[nodiscard]] bool success() const { return result == 0; }

  private:
    int result{};
    std::vector<int> ipiv{};
};
}

#endif