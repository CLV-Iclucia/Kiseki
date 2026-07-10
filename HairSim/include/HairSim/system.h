//
// Created by creeper on 23-8-9.
//

#ifndef KISEKI_HAIRSIM_INCLUDE_HAIRSIM_SYSTEM_H_
#define KISEKI_HAIRSIM_INCLUDE_HAIRSIM_SYSTEM_H_
#include <HairSim/hair-sim.h>
#include <HairSim/band-matrix.h>
#include <HairSim/hair.h>
#include <Core/properties.h>
#include <memory>
#include <vector>
#include <utility>

namespace hairsim {
struct VertexIdentifier {
  Index hairIndex;
  Index vertexIndex;
};

struct SystemConfig {
  Index nHairs = 0;
  Index nVerticesPerHair = 0;
  std::vector<VertexIdentifier> controlVertices{};

  [[nodiscard]] Index numVertices() const { return nHairs * nVerticesPerHair; }
};

struct System : core::NonCopyable {
  explicit System(SystemConfig config_, HairParams params_,
                  const std::vector<std::vector<Vec3d>>& ref_pos,
                  const std::vector<std::vector<Real>>& ref_theta) :
    config(std::move(config_)), params(std::move(params_)),
    q(config.numVertices()), qdot(config.numVertices()),
    rhs(config.numVertices()), matrix(config.numVertices()),
    hairs(config.nHairs) {
    for (int i = 0; i < config.nHairs; i++)
      hairs[i] = std::make_unique<Hair>(this, i,
                                        RefConfig(ref_pos[i], ref_theta[i]));
  }

  [[nodiscard]] Index numHairs() const {
    return config.nHairs;
  }

  [[nodiscard]] Index numVerticesPerHair() const {
    return config.nVerticesPerHair;
  }

  [[nodiscard]] Index numVertices() const {
    return config.nHairs * config.nVerticesPerHair;
  }

  SystemConfig config{};
  HairParams params{};
  VecXd q{};
  VecXd qdot{};
  VecXd rhs{};
  BandSquareMatrix<Real, 5> matrix;
  std::vector<std::unique_ptr<Hair>> hairs;
};
}
#endif // KISEKI_HAIRSIM_INCLUDE_HAIRSIM_HAIR_SYSTEM_H_