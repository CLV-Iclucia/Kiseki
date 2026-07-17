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

bool DofLayout::hasNonOverlappingRanges() const noexcept
{
  int expected_offset = 0;
  for (const auto& range : ranges) {
    if (range.scalarOffset != expected_offset || range.scalarCount < 0) {
      return false;
    }
    expected_offset += range.scalarCount;
  }
  return expected_offset == totalScalars;
}

}  // namespace ksk::runtime
