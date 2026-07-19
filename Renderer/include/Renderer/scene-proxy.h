#pragma once
#include <Core/core.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

namespace ksk::renderer {

struct MeshProxy {
  std::string name;
  std::vector<core::Vec3f> positions;
  std::vector<core::Vec3u> triangles;
  std::vector<core::Vec3f> normals;
  std::vector<core::Vec3f> colors;
  core::Vec3f objectColor{-1.0f};
};


  struct VolumeProxy
  {
    std::string name;

  };

struct WireframeProxy {
  std::string name;
  std::vector<core::Vec3f> positions;
  std::vector<core::Vec2u> edges;
  core::Vec3f color{1.0f, 1.0f, 1.0f};
};

struct ParticleProxy {
  std::string name;
  std::vector<core::Vec3f> positions;
  float radius = 0.01f;
  core::Vec3f color{0.2f, 0.5f, 1.0f};
};

struct CameraState {
  glm::vec3 position{0, 2, 5};
  glm::vec3 target{0, 0, 0};
  glm::vec3 up{0, 1, 0};
  float fov = 45.0f;        // degrees
  float nearPlane = 0.01f;
  float farPlane = 100.0f;
};

struct SceneProxy {
  std::vector<MeshProxy> meshes;
  std::vector<WireframeProxy> wireframes;
  std::vector<ParticleProxy> particles;
  std::vector<VolumeProxy> volumes;
  CameraState camera;
  float simulationTime = 0.0f;
  int frameIndex = 0;
};

inline void computeSmoothNormals(MeshProxy& mesh) {
  const auto& pos = mesh.positions;
  const auto& tris = mesh.triangles;

  mesh.normals.assign(pos.size(), core::Vec3f(0.0f));

  for (const auto& tri : tris) {
    core::Vec3f e1 = pos[tri.y] - pos[tri.x];
    core::Vec3f e2 = pos[tri.z] - pos[tri.x];
    core::Vec3f fn = glm::cross(e1, e2);  // area-weighted
    mesh.normals[tri.x] += fn;
    mesh.normals[tri.y] += fn;
    mesh.normals[tri.z] += fn;
  }

  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    n = (len > 1e-8f) ? n / len : core::Vec3f(0.0f, 1.0f, 0.0f);
  }
}

} // namespace ksk::renderer
