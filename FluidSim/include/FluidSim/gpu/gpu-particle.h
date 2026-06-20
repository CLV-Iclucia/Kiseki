// ============================================================================
// include/FluidSim/gpu/gpu-particle.h
// GPU particle data layout: SOA (Structure of Arrays)
//   - positions: float3 per particle (stored as 3 consecutive floats, no padding)
//   - velocities: float3 per particle (stored as 3 consecutive floats, no padding)
// Each particle occupies 12 bytes in positions buffer, 12 bytes in velocities buffer.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstddef>

namespace fluid::gpu {

// SOA layout constants
constexpr size_t kParticleFloatsPerVec3 = 3;  // x, y, z (tightly packed)
constexpr size_t kParticlePositionStride = kParticleFloatsPerVec3 * sizeof(float);  // 12 bytes
constexpr size_t kParticleVelocityStride = kParticleFloatsPerVec3 * sizeof(float);  // 12 bytes

/// Compute buffer size (bytes) for N particles' positions or velocities
inline constexpr size_t particleBufferSize(uint32_t numParticles) {
    return static_cast<size_t>(numParticles) * kParticleFloatsPerVec3 * sizeof(float);
}

} // namespace fluid::gpu
