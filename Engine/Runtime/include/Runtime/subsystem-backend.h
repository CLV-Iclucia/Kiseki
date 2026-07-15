#pragma once

namespace ksk::runtime {

class Subsystem;
class CpuSubsystemBackend;
class GpuSubsystemBackend;

class SubsystemBackend {
 public:
  virtual ~SubsystemBackend() = default;
  virtual void accept(Subsystem& subsystem) = 0;
};

class CpuSubsystemBackend final : public SubsystemBackend {
 public:
  void accept(Subsystem& subsystem) override;
};

class GpuSubsystemBackend final : public SubsystemBackend {
 public:
  void accept(Subsystem& subsystem) override;
};

}  // namespace ksk::runtime
