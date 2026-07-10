#include <fem/primitives/tet-mesh.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

namespace ksk::fem {
std::optional<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path) {
  // open file
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return std::nullopt;
  }
  std::string line;
  std::vector<Vector<Real, 3>> vertices;
  std::vector<Tetrahedron> indices;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string prefix;
    stream >> prefix;
    if (prefix == "v") {
      Real x, y, z;
      stream >> x >> y >> z;
      vertices.emplace_back(x, y, z);
    } else if (prefix == "t") {
      int x, y, z, w;
      stream >> x >> y >> z >> w;
      indices.emplace_back(x, y, z, w);
    }
  }
  file.close();
  // Auto-detect 0-based vs 1-based indexing:
  // If max index == numVertices, it's 1-based; convert to 0-based.
  if (!indices.empty()) {
    int maxIdx = 0;
    for (const auto& tet : indices)
      for (int k = 0; k < 4; k++)
        maxIdx = std::max(maxIdx, tet[k]);
    if (maxIdx == static_cast<int>(vertices.size())) {
      // 1-based → convert to 0-based
      for (auto& tet : indices)
        tet -= 1;
    }
  }
  return std::make_optional<TetMesh>(vertices, indices);
}

TetMesh TetMesh::static_deserialize(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary");
  const auto &dict = json.as<core::JsonDict>();

  if (dict.contains("path")) {
    const auto &mesh =
        readTetMeshFromTOBJ(dict.at("path").as<std::string>());
    if (!mesh)
      throw std::runtime_error("Failed to load mesh");
    return *mesh;
  }

  if (dict.contains("vertices") && dict.contains("tets")) {
    if (dict.contains("velocities"))
      return TetMesh{
          core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("vertices")),
          core::deserialize<std::vector<Tetrahedron>>(dict.at("tets")),
          core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("velocities"))};
    return TetMesh{
        core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("vertices")),
        core::deserialize<std::vector<Tetrahedron>>(dict.at("tets"))};
  }

  throw std::runtime_error("Expected either a path or vertices and tets");
}

void TetMesh::computeSurface() {
  struct TriangleCmp {
    bool operator()(const Triangle &a, const Triangle &b) const {
      if (a.x != b.x)
        return a.x < b.x;
      if (a.y != b.y)
        return a.y < b.y;
      return a.z < b.z;
    }
  };

  // Key: sorted triangle (for identifying shared faces)
  // Value: oriented triangle (correct outward-normal winding)
  std::map<Triangle, Triangle, TriangleCmp> face_map{};

  auto sortTriangle = [](const Triangle &t) -> Triangle {
    std::array<int, 3> s = {t[0], t[1], t[2]};
    std::ranges::sort(s);
    return Triangle{s[0], s[1], s[2]};
  };

  for (const auto &tet : tets) {
    // For a positively-oriented tet (v0, v1, v2, v3) where
    // det([v1-v0, v2-v0, v3-v0]) > 0, the outward-normal windings are:
    Triangle outwardFaces[4] = {
        {tet[1], tet[2], tet[3]},  // opposite v0
        {tet[0], tet[3], tet[2]},  // opposite v1
        {tet[0], tet[1], tet[3]},  // opposite v2
        {tet[0], tet[2], tet[1]},  // opposite v3
    };

    for (const auto &face : outwardFaces) {
      auto key = sortTriangle(face);
      if (face_map.contains(key))
        face_map.erase(key);      // internal face (shared by two tets)
      else
        face_map[key] = face;     // boundary face with correct orientation
    }
  }

  surfaces.clear();
  surfaces.reserve(face_map.size());
  for (const auto &[_, orientedFace] : face_map)
    surfaces.push_back(orientedFace);
}

void TetMesh::computeSurfaceEdges() {
  struct EdgeCmp {
    bool operator()(const Edge &a, const Edge &b) const {
      if (a.x != b.x)
        return a.x < b.x;
      return a.y < b.y;
    }
  };
  std::set<Edge, EdgeCmp> edge_set{};
  for (auto surface : surfaces) {
    if (surface[0] < surface[1] && !edge_set.contains({surface[0], surface[1]}))
      edge_set.insert({surface[0], surface[1]});
    if (surface[1] < surface[2] && !edge_set.contains({surface[1], surface[2]}))
      edge_set.insert({surface[1], surface[2]});
    if (surface[2] < surface[0] && !edge_set.contains({surface[2], surface[0]}))
      edge_set.insert({surface[2], surface[0]});
  }
  surfaceEdges.resize(static_cast<int>(edge_set.size()));
  int i = 0;
  for (const auto &edge : edge_set) {
    surfaceEdges[i] = {edge.x, edge.y};
    i++;
  }
}
} // namespace fem
