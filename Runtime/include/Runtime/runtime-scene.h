#pragma once

#include <cstdint>
#include <vector>

#include <Runtime/contact-table.h>
#include <Runtime/dof-layout.h>
#include <Runtime/geometry-table.h>
#include <Runtime/solver-plan.h>

namespace ksk::runtime {

enum class SubsystemType : std::uint16_t {
  TetFEM,
  DERod,
  AffineBody,
  Custom,
};

struct SubsystemBatch {
  SubsystemType type = SubsystemType::Custom;
  int firstSubsystem = 0;
  int subsystemCount = 0;
  int qOffset = 0;
  int qCount = 0;
  int geometryOffset = 0;
  int geometryCount = 0;
  int auxOffset = 0;
  int auxCount = 0;
};

struct BufferLayout {
  int qScalars = 0;
  int qdotScalars = 0;
  int geometryPoints = 0;
};

struct RuntimeScene {
  DofLayout dofs;
  std::vector<SubsystemBatch> subsystemBatches;
  GeometryTable geometry;
  ContactTable contacts;
  SolverPlan solverPlan;
  BufferLayout buffers;

  void refreshBufferLayout() noexcept;
};

}  // namespace ksk::runtime
