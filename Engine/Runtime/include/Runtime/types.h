#pragma once

#include <cstddef>

namespace ksk::runtime {

using Real = double;

using SubsystemId = int;
using GeometryPointId = int;

using ObjectId = int;
using ObjectTypeId = const void*;

template <typename T>
inline ObjectTypeId elementTypeId() noexcept
{
  static const int token = 0;
  return ObjectTypeId{&token};
}

struct SubsystemRange {
  int first = 0;
  int count = 0;
};

}  // namespace ksk::runtime
