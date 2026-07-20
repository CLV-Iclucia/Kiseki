#include <Runtime/geometry-transfer.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace ksk::runtime {
namespace {

TEST(GeometryTransfer, RemapsTopologyAfterExistingGeometry)
{
  GlobalGeometryManager geometry;

  const std::vector<glm::dvec3> first_vertices{
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
      glm::dvec3{0.0, 0.0, 1.0},
  };
  const std::vector<std::array<int, 3>> first_triangles{
      std::array<int, 3>{0, 1, 2},
  };
  const std::vector<std::array<int, 4>> first_tets{
      std::array<int, 4>{0, 1, 2, 3},
  };
  [[maybe_unused]] const GeometryTransferMap first = transferGeometry(
      geometry,
      GeometryTransferInput{
          .sourceObject = 10,
          .subsystem = 0,
          .vertices = first_vertices,
          .triangles = first_triangles,
          .tets = first_tets,
      });

  const std::vector<glm::dvec3> second_vertices{
      glm::dvec3{10.0, 0.0, 0.0},
      glm::dvec3{11.0, 0.0, 0.0},
      glm::dvec3{10.0, 1.0, 0.0},
      glm::dvec3{10.0, 0.0, 1.0},
  };
  const std::vector<std::array<int, 2>> second_edges{
      std::array<int, 2>{0, 3},
  };
  const std::vector<std::array<int, 3>> second_triangles{
      std::array<int, 3>{0, 2, 3},
  };
  const std::vector<std::array<int, 4>> second_tets{
      std::array<int, 4>{3, 2, 1, 0},
  };
  const GeometryTransferMap second = transferGeometry(
      geometry,
      GeometryTransferInput{
          .sourceObject = 11,
          .subsystem = 1,
          .localSampleOffset = 8,
          .vertices = second_vertices,
          .edges = second_edges,
          .triangles = second_triangles,
          .tets = second_tets,
      });

  ASSERT_EQ(second.points.first, 4);
  EXPECT_EQ(second.points.count, 4);
  EXPECT_EQ(second.edges.count, 1);
  EXPECT_EQ(second.triangles.count, 1);
  EXPECT_EQ(second.tets.count, 1);
  EXPECT_EQ(second.localToGlobalPoint[0], 4);
  EXPECT_EQ(second.localToGlobalPoint[3], 7);
  EXPECT_EQ(geometry.pointRef(second.localToGlobalPoint[0]).localIndex, 8);

  const auto triangle = geometry.globalTriangle(second.localToGlobalTriangle[0]);
  EXPECT_EQ(triangle[0], 4);
  EXPECT_EQ(triangle[1], 6);
  EXPECT_EQ(triangle[2], 7);

  const auto tet = geometry.globalTet(second.localToGlobalTet[0]);
  EXPECT_EQ(tet[0], 7);
  EXPECT_EQ(tet[1], 6);
  EXPECT_EQ(tet[2], 5);
  EXPECT_EQ(tet[3], 4);
}

TEST(GeometryTransfer, RejectsInvalidLocalTopology)
{
  GlobalGeometryManager geometry;

  const std::vector<glm::dvec3> vertices{
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
  };
  const std::vector<std::array<int, 3>> triangles{
      std::array<int, 3>{0, 2, 3},
  };

  EXPECT_THROW([[maybe_unused]] const GeometryTransferMap map = transferGeometry(
                   geometry,
                   GeometryTransferInput{
                       .sourceObject = 2,
                       .subsystem = 0,
                       .vertices = vertices,
                       .triangles = triangles,
                   }),
               std::out_of_range);
}

}  // namespace
}  // namespace ksk::runtime
