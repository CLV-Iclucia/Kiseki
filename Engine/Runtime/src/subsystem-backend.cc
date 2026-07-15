#include <Runtime/subsystem-backend.h>

#include <Runtime/subsystem.h>

namespace ksk::runtime {

void CpuSubsystemBackend::accept(Subsystem& subsystem)
{
  subsystem.visit(*this);
}

void GpuSubsystemBackend::accept(Subsystem& subsystem)
{
  subsystem.visit(*this);
}

}  // namespace ksk::runtime
