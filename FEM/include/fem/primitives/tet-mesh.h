//
// Created by creeper on 5/24/24.
//

#pragma once
#include <Core/deserializer.h>
#include <fem/simplex.h>
#include <fem/types.h>
#include <filesystem>
#include <optional>
#include <span>

namespace sim::fem {
struct TetMesh {
  TetMesh() = default;
  TetMesh(const std::vector<Vector<Real, 3>> &vertices,
          const std::vector<Tetrahedron> &tets,
          const std::vector<Vector<Real, 3>> &velocities = {},
          const std::vector<Vector<Real, 3>> &initialPositions = {})
      : vertices(vertices), tets(tets), velocities(velocities),
        initialPositions(initialPositions) {
    ensurePositiveOrientation();
    computeSurface();
    computeSurfaceEdges();
  }
  std::vector<Tetrahedron> tets{};
  std::vector<Triangle> surfaces{};
  std::vector<Edge> surfaceEdges{};
  static TetMesh static_deserialize(const core::JsonNode &json);
  [[nodiscard]] std::span<const Triangle> surfaceView() const {
    return std::span{surfaces};
  }
  [[nodiscard]] std::span<const Edge> surfaceEdgeView() const {
    return std::span{surfaceEdges};
  }
  /// Commit mesh data into the system vectors.
  /// - X (rest shape) is always set from `vertices` (the rest configuration).
  /// - x (current config) is set from `initialPositions` if provided, otherwise from `vertices`.
  /// - xdot is set from `velocities` if provided, otherwise zero.
  void commit(SubVector<Real> x, SubVector<Real> xdot, SubVector<Real> X) {
    for (int i = 0; i < static_cast<int>(vertices.size()); i++) {
      // X = rest shape (always from vertices)
      X.segment<3>(i * 3) = vertices[i];

      // x = initial position (from initialPositions if given, else rest)
      if (!initialPositions.empty())
        x.segment<3>(i * 3) = initialPositions[i];
      else
        x.segment<3>(i * 3) = vertices[i];

      // xdot = initial velocity
      if (!velocities.empty())
        xdot.segment<3>(i * 3) = velocities[i];
      else
        xdot.segment<3>(i * 3) = Vector<Real, 3>::Zero();
    }
    vertices.clear();
    initialPositions.clear();
    transitionToCommitted();
  }
  const std::vector<Vector<Real, 3>> &getVertices() const {
    if (committed)
      throw std::runtime_error("Mesh vertices are committed and cannot be accessed");
    return vertices;
  }
  /// Get initial positions (deformed configuration at t=0).
  /// Returns rest vertices if no initial positions were specified.
  const std::vector<Vector<Real, 3>> &getInitialPositions() const {
    if (committed)
      throw std::runtime_error("Mesh data is committed and cannot be accessed");
    if (!initialPositions.empty())
      return initialPositions;
    return vertices;
  }

private:
  // these will be committed and cleared so they cannot be accessed after commit
  std::vector<Vector<Real, 3>> vertices{};       // rest shape (X)
  std::vector<Vector<Real, 3>> velocities{};     // initial velocity
  std::vector<Vector<Real, 3>> initialPositions{}; // initial deformed positions (x at t=0); empty → use vertices
  bool committed{false};
  void transitionToCommitted() {
    committed = true;
  }
  void computeSurface();
  void computeSurfaceEdges();

  /// 确保所有四面体正定向（det > 0）。若为负则交换顶点 0 和 1。
  void ensurePositiveOrientation() {
    for (auto &tet : tets) {
      Matrix<Real, 3, 3> edges;
      for (int j = 0; j < 3; j++)
        edges.col(j) = vertices[tet[j + 1]] - vertices[tet[0]];
      if (edges.determinant() < 0)
        std::swap(tet[0], tet[1]);
    }
  }
};

std::optional<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path);
} // namespace sim::fem