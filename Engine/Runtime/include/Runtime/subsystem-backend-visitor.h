#pragma once

namespace ksk::runtime {

}  // namespace ksk::runtime

namespace ksk::der {

class DERSubsystem;

}  // namespace ksk::der

namespace ksk::runtime {

class SubsystemBackendVisitor {
 public:
  virtual ~SubsystemBackendVisitor() = default;
  virtual void visit(ksk::der::DERSubsystem& subsystem) = 0;
};

class CpuSubsystemBackendVisitor : public virtual SubsystemBackendVisitor {
 public:
  ~CpuSubsystemBackendVisitor() override = default;
};

class GpuSubsystemBackendVisitor : public virtual SubsystemBackendVisitor {
 public:
  ~GpuSubsystemBackendVisitor() override = default;
};

}  // namespace ksk::runtime
