#include <Runtime/geometry-table.h>

namespace ksk::runtime {

GeometryPointId GeometryTable::appendPoint(SubsystemId subsystem,
                                           int localSampleId,
                                           const glm::dvec3& position)
{
  GeometryPointId id{static_cast<int>(points.size())};
  points.push_back(GeometryPoint{
      .id = id,
      .subsystem = subsystem,
      .localSampleId = localSampleId,
      .x = position,
  });
  return id;
}

int GeometryTable::appendEdge(GeometryPointId p0, GeometryPointId p1, Real radius)
{
  const int id = static_cast<int>(edges.size());
  edges.push_back(GeometryEdge{
      .id = id,
      .p0 = p0,
      .p1 = p1,
      .radius = radius,
  });
  return id;
}

int GeometryTable::appendTriangle(GeometryPointId p0, GeometryPointId p1, GeometryPointId p2)
{
  const int id = static_cast<int>(triangles.size());
  triangles.push_back(GeometryTriangle{
      .id = id,
      .p0 = p0,
      .p1 = p1,
      .p2 = p2,
  });
  return id;
}

bool GeometryTable::contains(GeometryPointId point) const noexcept
{
  return point.value >= 0 && point.value < static_cast<int>(points.size());
}

}  // namespace ksk::runtime
