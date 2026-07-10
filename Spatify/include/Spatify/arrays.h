//
// Created by creeper on 10/24/23.
//

#ifndef KISEKI_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_
#define KISEKI_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_

#include <Spatify/types.h>
#include <Spatify/parallel.h>
#include <Spatify/hash.h>
#include <vector>

namespace spatify {
enum class PaddingMode {
  Around,
};

template <typename T, int Padding, PaddingMode = PaddingMode::Around>
class GhostArray2D {
  public:
    GhostArray2D() = default;
    GhostArray2D(int width, int height)
      : m_width(width), m_height(height) {
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding));
    }
    T& operator()(int i, int j) {
      return m_data[(i + Padding) + (j + Padding) * (m_width + 2 * Padding)];
    }
    const T& operator()(int i, int j) const {
      return m_data[(i + Padding) + (j + Padding) * (m_width + 2 * Padding)];
    }
    T& operator()(const Vec2i& p) {
      return m_data[(p.x + Padding) + (p.y + Padding) * (
                      m_width + 2 * Padding)];
    }
    const T& operator()(const Vec2i& p) const {
      return m_data[(p.x + Padding) + (p.y + Padding) * (
                      m_width + 2 * Padding)];
    }
    T& operator[](int i) { return m_data[i]; }
    const T& operator[](int i) const { return m_data[i]; }
    int width() const { return m_width; }
    int height() const { return m_height; }
    int size() const { return m_data.size(); }
    void resize(int width, int height) {
      m_width = width;
      m_height = height;
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding));
    }
    void resize(int width, int height, const T& value) {
      m_width = width;
      m_height = height;
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding), value);
    }
    void fill(const T& non_padding_value, const T& padding_value) {
      for (int i = -Padding; i < m_width + Padding; i++) {
        for (int j = -Padding; j < m_height + Padding; j++) {
          if (i < 0 || i >= m_width || j < 0 || j >= m_height) {
            (*this)(i, j) = padding_value;
          } else {
            (*this)(i, j) = non_padding_value;
          }
        }
      }
    }
    template <typename Func>
    void forEach(Func&& func) {
      for (int i = 0; i < m_width; i++)
        for (int j = 0; j < m_height; j++)
          func(i, j);
    }
    template <typename Func>
    void parallelForEach(Func&& func) {
      parallel_for(0, m_width, [&](int i) {
        for (int j = 0; j < m_height; j++)
          func(i, j);
      });
    }
    template <typename Func>
    void forEach(Func&& func) const {
      tbb::parallel_for(0, m_width, [&](int i) {
        for (int j = 0; j < m_height; j++)
          func(i, j);
      });
    }
    void swap(GhostArray2D& other) {
      m_data.swap(other.m_data);
      std::swap(m_width, other.m_width);
      std::swap(m_height, other.m_height);
    }
    void clear() {
      m_data.clear();
      m_width = 0;
      m_height = 0;
    }
    void reserve(int n) { m_data.reserve(n); }
    void shrink_to_fit() { m_data.shrink_to_fit(); }
    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }
    std::vector<T>& raw() { return m_data; }
    const std::vector<T>& raw() const { return m_data; }

  private:
    std::vector<T> m_data;
    int m_width = 0, m_height = 0;
};

template <typename T, int Padding, SpatialHashFunction3D Hash = LinearHashXYZ,
          PaddingMode = PaddingMode::Around>
class GhostArray3D {
  public:
    GhostArray3D() = default;
    GhostArray3D(int width, int height, int depth)
      : m_width(width), m_height(height), m_depth(depth), hash(
            width + 2 * Padding, height + 2 * Padding,
            depth + 2 * Padding) {
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
                    (depth + 2 * Padding));
    }
    explicit GhostArray3D(const Vec3i& size)
      : m_width(size.x), m_height(size.y), m_depth(size.z), hash(
            size.x + 2 * Padding, size.y + 2 * Padding,
            size.z + 2 * Padding) {
      m_data.resize((size.x + 2 * Padding) * (size.y + 2 * Padding) *
                    (size.z + 2 * Padding));
    }
    T& operator()(int i, int j, int k) {
      return m_data[hash(i + Padding, j + Padding, k + Padding)];
    }
    const T& operator()(int i, int j, int k) const {
      return m_data[hash(i + Padding, j + Padding, k + Padding)];
    }
    T& operator()(const Vec3i& idx) {
      return m_data[hash(idx.x + Padding, idx.y + Padding, idx.z + Padding)];
    }
    const T& operator()(const Vec3i& idx) const {
      return m_data[hash(idx.x + Padding, idx.y + Padding, idx.z + Padding)];
    }
    T& at(int i, int j, int k) {
      return m_data[hash(i + Padding, j + Padding, k + Padding)];
    }
    const T& at(int i, int j, int k) const {
      return m_data[hash(i + Padding, j + Padding, k + Padding)];
    }
    T& at(const Vec3i& idx) {
      return m_data[hash(idx.x + Padding, idx.y + Padding, idx.z + Padding)];
    }
    const T& at(const Vec3i& idx) const {
      return m_data[hash(idx.x + Padding, idx.y + Padding, idx.z + Padding)];
    }
    int width() const { return m_width; }
    int height() const { return m_height; }
    int depth() const { return m_depth; }
    int size() const { return m_data.size(); }
    void resize(int width, int height, int depth) {
      m_width = width;
      m_height = height;
      m_depth = depth;
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
                    (depth + 2 * Padding));
    }
    void resize(int width, int height, int depth, const T& value) {
      m_width = width;
      m_height = height;
      m_depth = depth;
      m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
                    (depth + 2 * Padding), value);
    }
    void fill(const T& value) {
      std::fill(m_data.begin(), m_data.end(), value);
    }
    void fill(const T& non_padding_value, const T& padding_value) {
      if constexpr (Padding == 0) {
        memset(m_data.data(), non_padding_value,
               m_width * m_height * m_depth * sizeof(T));
      } else {
        for (int i = -Padding; i < m_width + Padding; i++) {
          for (int j = -Padding; j < m_height + Padding; j++) {
            for (int k = -Padding; k < m_depth + Padding; k++) {
              if (i < 0 || i >= m_width || j < 0 || j >= m_height || k < 0 || k
                  >= m_depth) {
                (*this)(i, j, k) = padding_value;
              } else {
                (*this)(i, j, k) = non_padding_value;
              }
            }
          }
        }
      }
    }
    template <typename Func>
    void forEach(Func&& func) {
      for (int i = 0; i < m_width; i++)
        for (int j = 0; j < m_height; j++)
          for (int k = 0; k < m_depth; k++)
            func(i, j, k);
    }
    template <typename Func>
    void forEach(Func&& func) const {
      for (int i = 0; i < m_width; i++)
        for (int j = 0; j < m_height; j++)
          for (int k = 0; k < m_depth; k++)
            func(i, j, k);
    }
    template <typename Func>
    void forEachReversed(Func&& func) {
      for (int i = m_width - 1; i >= 0; i--)
        for (int j = m_height - 1; j >= 0; j--)
          for (int k = m_depth - 1; k >= 0; k--)
            func(i, j, k);
    }
    template <typename Func>
    void forEachReversed(Func&& func) const {
      for (int i = m_width - 1; i >= 0; i++)
        for (int j = m_height - 1; j >= 0; j++)
          for (int k = m_depth - 1; k >= 0; k++)
            func(i, j, k);
    }
    template <typename Func>
    void parallelForEach(Func&& func) const {
      parallel_for(0, m_width, [&](int i) {
        for (int j = 0; j < m_height; j++)
          for (int k = 0; k < m_depth; k++)
            func(i, j, k);
      });
    }
    template <typename Func>
    void parallelForEach(Func&& func) {
      parallel_for(0, m_width, [&](int i) {
        for (int j = 0; j < m_height; j++)
          for (int k = 0; k < m_depth; k++)
            func(i, j, k);
      });
    }
    void copyFrom(const GhostArray3D& other) {
      assert(m_width == other.m_width && m_height == other.m_height &&
          m_depth == other.m_depth);
      memcpy(m_data.data(), other.m_data.data(), m_data.size() * sizeof(T));
    }
    void swap(GhostArray3D& other) {
      m_data.swap(other.m_data);
      std::swap(m_width, other.m_width);
      std::swap(m_height, other.m_height);
      std::swap(m_depth, other.m_depth);
    }
    void clear() {
      m_data.clear();
      m_width = 0;
      m_height = 0;
      m_depth = 0;
    }
    void reserve(int n) { m_data.reserve(n); }
    void shrink_to_fit() { m_data.shrink_to_fit(); }
    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }
    std::vector<T>& raw() { return m_data; }
    const std::vector<T>& raw() const { return m_data; }

  private:
    std::vector<T> m_data;
    int m_width = 0, m_height = 0, m_depth = 0;
    Hash hash;
};
template <typename T>
using Array2D = GhostArray2D<T, 0>;
template <typename T>
using Array3D = GhostArray3D<T, 0>;
}

#endif //KISEKI_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_