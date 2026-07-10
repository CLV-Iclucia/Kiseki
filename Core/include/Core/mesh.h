//
// Created by creeper on 23-9-1.
//

#pragma once
#include <Core/core.h>
#include <string>
namespace ksk::core {
struct Mesh {
  int triangleCount;
  std::vector<Vec3d> vertices;
  std::vector<Vec3d> normals;
  std::vector<uint> indices;
};

bool myLoadObj(const std::string& path, Mesh* mesh);
bool exportObj(const std::string& path, const Mesh& mesh);
}// namespace core