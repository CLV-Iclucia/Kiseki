#include <Runtime/runtime-simulation.h>

#include <algorithm>
#include <utility>

namespace ksk::runtime {

RuntimeSimulation::RuntimeSimulation(
    std::vector<std::unique_ptr<Subsystem>> subsystems,
    GlobalSolverConfig solver,
    glm::dvec3 gravity)
    : RuntimeSimulation(std::move(subsystems), {}, solver, gravity)
{
}

RuntimeSimulation::RuntimeSimulation(
    std::vector<std::unique_ptr<Subsystem>> subsystems,
    std::vector<std::string> subsystemTypeNames,
    GlobalSolverConfig solver,
    glm::dvec3 gravity)
    : subsystems_(std::move(subsystems))
    , subsystem_type_names_(std::move(subsystemTypeNames))
    , gravity_(gravity)
{
  scene_.solverConfig = solver;
  scene_.gravity = gravity_;
  rebuild();
}

void RuntimeSimulation::rebuild()
{
  scene_.dofs = {};
  scene_.geometry = {};
  scene_.contacts = {};
  scene_.subsystemBatches.clear();
  scene_.gravity = gravity_;

  int total_scalars = 0;
  for (int index = 0; index < static_cast<int>(subsystems_.size()); ++index) {
    const auto& subsystem = subsystems_[index];
    const DofRange range = subsystem->dofRange();
    const int geometry_offset = scene_.geometry.pointCount();
    scene_.dofs.ranges.push_back(range);
    total_scalars =
        std::max(total_scalars, range.scalarOffset + range.scalarCount);
    subsystem->declareGeometry(scene_.geometry);

    const int geometry_count =
        scene_.geometry.pointCount() - geometry_offset;
    const std::string type_name =
        index < static_cast<int>(subsystem_type_names_.size())
            ? subsystem_type_names_[index]
            : std::string{};
    scene_.subsystemBatches.push_back({
        .type = type_name,
        .firstSubsystem = index,
        .subsystemCount = 1,
        .qOffset = range.scalarOffset,
        .qCount = range.scalarCount,
        .geometryOffset = geometry_offset,
        .geometryCount = geometry_count,
    });
  }
  scene_.dofs.totalScalars = total_scalars;
  scene_.refreshBufferLayout();

  q_ = DofBuffer::CPU(scene_.dofs.totalScalars);
  qdot_ = DofBuffer::CPU(scene_.dofs.totalScalars);
  for (const auto& subsystem : subsystems_) {
    subsystem->writeState(q_, qdot_);
  }
}

std::vector<std::unique_ptr<Subsystem>>& RuntimeSimulation::subsystems() noexcept
{
  return subsystems_;
}

const std::vector<std::unique_ptr<Subsystem>>& RuntimeSimulation::subsystems()
    const noexcept
{
  return subsystems_;
}

}  // namespace ksk::runtime
