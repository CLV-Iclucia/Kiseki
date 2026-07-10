#pragma once

namespace ksk::runtime {

using Real = double;

struct SubsystemId {
  int value = -1;

  [[nodiscard]] constexpr bool isValid() const noexcept { return value >= 0; }
};

struct GeometryPointId {
  int value = -1;

  [[nodiscard]] constexpr bool isValid() const noexcept { return value >= 0; }
};

struct SubsystemRange {
  int first = 0;
  int count = 0;
};

[[nodiscard]] constexpr bool operator==(SubsystemId lhs, SubsystemId rhs) noexcept
{
  return lhs.value == rhs.value;
}

[[nodiscard]] constexpr bool operator!=(SubsystemId lhs, SubsystemId rhs) noexcept
{
  return !(lhs == rhs);
}

[[nodiscard]] constexpr bool operator==(GeometryPointId lhs, GeometryPointId rhs) noexcept
{
  return lhs.value == rhs.value;
}

[[nodiscard]] constexpr bool operator!=(GeometryPointId lhs, GeometryPointId rhs) noexcept
{
  return !(lhs == rhs);
}

}  // namespace ksk::runtime
