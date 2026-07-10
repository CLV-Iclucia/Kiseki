//
// Created by creeper on 23-9-1.
//
#include <Core/mesh.h>
#include <Core/utils.h>

#include <format>
#include <iostream>
#include <sstream>
#include <fstream>

namespace ksk::core {
bool myLoadObj(const std::string& path, Mesh* mesh) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return false;
  }
  std::string line;
  std::vector<Vec3d> normals;
  std::vector<int> normal_cnt;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string prefix;
    stream >> prefix;
    if (prefix == "v") {
      Real x, y, z;
      stream >> x >> y >> z;
      mesh->vertices.emplace_back(x, y, z);
      mesh->normals.emplace_back(0.0, 0.0, 0.0);
      normal_cnt.emplace_back(0);
    } else if (prefix == "f") {
      for (int i = 0; i < 3; ++i) {
        std::string data;
        stream >> data;
        size_t pos = data.find("//");
        if (pos == std::string::npos) {
          int idx = std::stoi(data) - 1;
          mesh->indices.emplace_back(idx);
          continue;
        }
        int idx = std::stoi(data.substr(0, pos)) - 1;
        int normal_idx = std::stoi(data.substr(pos + 2, data.length())) - 1;
        mesh->indices.emplace_back(idx);
        mesh->normals[idx] += normals[normal_idx];
        normal_cnt[idx]++;
      }
    } else if (prefix == "vn") {
      Real x, y, z;
      stream >> x >> y >> z;
      normals.emplace_back(x, y, z);
    }
  }
  assert(mesh->indices.size() % 3 == 0);
  mesh->triangleCount = mesh->indices.size() / 3;
  for (int i = 0; i < mesh->normals.size(); i++) {
    if (normal_cnt[i] == 0)
      continue;
    mesh->normals[i] /= normal_cnt[i];
    mesh->normals[i] = normalize(mesh->normals[i]);
  }
  file.close();
  return true;
}

bool exportObj(const std::string& filename, const Mesh& mesh) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    file.open(filename, std::ios::out);
    if (!file.is_open()) {
      std::cerr << std::format("Error: Unable to create file {}", filename) <<
          std::endl;
      return false;
    }
  }

  for (const auto& vertex : mesh.vertices)
    file << std::format("v {} {} {}\n", vertex.x, vertex.y, vertex.z);
  for (const auto& normal : mesh.normals)
    file << std::format("vn {} {} {}\n", normal.x, normal.y, normal.z);

  for (size_t i = 0; i < mesh.indices.size(); i += 3) {
    file << "f ";
    for (int j = 0; j < 3; ++j) {
      file << std::format("{}//{} ", mesh.indices[i + j] + 1,
                          mesh.indices[i + j] + 1);
    }
    file << "\n";
  }
  return true;
}
}