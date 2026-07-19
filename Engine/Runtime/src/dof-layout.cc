#include <Runtime/dof-layout.h>

#include <algorithm>

namespace ksk::runtime {

DofRange DofLayout::appendRange(SubsystemId subsystem, int scalarCount, int blockSize)
{
  DofRange range{
      .subsystem = subsystem,
      .scalarOffset = totalScalars,
      .scalarCount = scalarCount,
      .blockSize = blockSize,
  };
  ranges.push_back(range);
  totalScalars += scalarCount;
  return range;
}

std::optional<DofRange> DofLayout::findRange(SubsystemId subsystem) const
{
  const auto it = std::find_if(ranges.begin(), ranges.end(), [subsystem](const DofRange& range) {
    return range.subsystem == subsystem;
  });
  if (it == ranges.end()) {
    return std::nullopt;
  }
  return *it;
}

}  // namespace ksk::runtime
