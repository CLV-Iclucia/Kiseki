#pragma once

#include <Runtime/buffers.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/subsystem.h>

#include <memory>
#include <string>
#include <vector>

namespace ksk::runtime {

class RuntimeSimulation {
 public:
  RuntimeSimulation() = default;
  explicit RuntimeSimulation(std::vector<std::unique_ptr<Subsystem>> subsystems,
                             GlobalSolverConfig solver = {},
                             glm::dvec3 gravity = glm::dvec3(0.0, -9.81, 0.0));
  RuntimeSimulation(std::vector<std::unique_ptr<Subsystem>> subsystems,
                    std::vector<std::string> subsystemTypeNames,
                    GlobalSolverConfig solver = {},
                    glm::dvec3 gravity = glm::dvec3(0.0, -9.81, 0.0));
  RuntimeSimulation(const RuntimeSimulation&) = delete;
  RuntimeSimulation& operator=(const RuntimeSimulation&) = delete;
  RuntimeSimulation(RuntimeSimulation&&) noexcept = default;
  RuntimeSimulation& operator=(RuntimeSimulation&&) noexcept = default;

  void rebuild();

  [[nodiscard]] RuntimeScene& scene() noexcept { return scene_; }
  [[nodiscard]] const RuntimeScene& scene() const noexcept { return scene_; }
  [[nodiscard]] DofBuffer& q() noexcept { return q_; }
  [[nodiscard]] const DofBuffer& q() const noexcept { return q_; }
  [[nodiscard]] DofBuffer& qdot() noexcept { return qdot_; }
  [[nodiscard]] const DofBuffer& qdot() const noexcept { return qdot_; }
  [[nodiscard]] std::vector<std::unique_ptr<Subsystem>>& subsystems() noexcept;
  [[nodiscard]] const std::vector<std::unique_ptr<Subsystem>>& subsystems()
      const noexcept;

 private:
  RuntimeScene scene_;
  std::vector<std::unique_ptr<Subsystem>> subsystems_;
  std::vector<std::string> subsystem_type_names_;
  glm::dvec3 gravity_{0.0, -9.81, 0.0};
  DofBuffer q_;
  DofBuffer qdot_;
};

}  // namespace ksk::runtime
