#pragma once
#include <Renderer/scene-proxy.h>
#include <FluidSim/fluid-backend.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace ksk::renderer {

/// 从 FluidBackend 构建渲染帧。
///
/// 与 FEM 端的 buildSceneProxyFromSystem() 完全对等：
///   - FEM:   system.x (double) → MeshProxy (float)
///   - Fluid: backend.readbackParticles (double) → ParticleProxy (float)
///
/// 调用方只需：
///   app.buildProxy = [&](int step) {
///       return buildSceneProxyFromFluid(backend, step);
///   };
///
inline std::unique_ptr<SceneProxy> buildSceneProxyFromFluid(
    fluid::FluidBackend& backend, int frameIndex,
    float particleRadius = 0.005f,
    core::Vec3f particleColor = {0.15f, 0.45f, 0.95f})
{
    // 1. Readback from backend (CPU or GPU → CPU copy)
    fluid::FluidFrame frame;
    backend.readbackParticles(frame);

    // 2. Build proxy
    auto proxy = std::make_unique<SceneProxy>();
    proxy->frameIndex = frameIndex;

    ParticleProxy particles;
    particles.name   = "fluid_particles";
    particles.radius = particleRadius;
    particles.color  = particleColor;

    // double → float 位置转换 (与 FEM 端 dvec3→vec3 一致)
    particles.positions.resize(frame.particlePositions.size());
    for (size_t i = 0; i < frame.particlePositions.size(); ++i) {
        const auto& p = frame.particlePositions[i];
        particles.positions[i] = core::Vec3f(
            static_cast<float>(p.x),
            static_cast<float>(p.y),
            static_cast<float>(p.z));
    }

    proxy->particles.push_back(std::move(particles));
    return proxy;
}

/// 带碰撞体网格的完整版本。
/// colliderMesh 如果提供，会作为半透明灰色 MeshProxy 加入场景。
inline std::unique_ptr<SceneProxy> buildSceneProxyFromFluid(
    fluid::FluidBackend& backend, int frameIndex,
    const fluid::Mesh* colliderMesh,
    float particleRadius = 0.005f,
    core::Vec3f particleColor = {0.15f, 0.45f, 0.95f})
{
    auto proxy = buildSceneProxyFromFluid(backend, frameIndex,
                                          particleRadius, particleColor);

    // 可选：添加碰撞体作为静态网格
    if (colliderMesh && colliderMesh->triangleCount > 0) {
        MeshProxy collider;
        collider.name = "collider";
        collider.objectColor = core::Vec3f(0.6f, 0.6f, 0.6f);  // 灰色

        collider.positions.resize(colliderMesh->vertices.size());
        for (size_t i = 0; i < colliderMesh->vertices.size(); ++i) {
            collider.positions[i] = core::Vec3f(
                static_cast<float>(colliderMesh->vertices[i].x),
                static_cast<float>(colliderMesh->vertices[i].y),
                static_cast<float>(colliderMesh->vertices[i].z));
        }

        size_t triCount = colliderMesh->indices.size() / 3;
        collider.triangles.resize(triCount);
        for (size_t i = 0; i < triCount; ++i) {
            collider.triangles[i] = core::Vec3u(
                colliderMesh->indices[i * 3 + 0],
                colliderMesh->indices[i * 3 + 1],
                colliderMesh->indices[i * 3 + 2]);
        }

        computeSmoothNormals(collider);
        proxy->meshes.push_back(std::move(collider));
    }

    return proxy;
}

} // namespace ksk::renderer
