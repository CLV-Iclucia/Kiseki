#include <Runtime/runtime-scene.h>

#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

TEST(RuntimeLayout, AppendsNonOverlappingDofRanges)
{
  DofLayout layout;

  const auto fem_range = layout.appendRange(SubsystemId{0}, 12, 3);
  const auto hair_range = layout.appendRange(SubsystemId{1}, 7, 0);

  EXPECT_EQ(fem_range.scalarOffset, 0);
  EXPECT_EQ(fem_range.scalarCount, 12);
  EXPECT_EQ(hair_range.scalarOffset, 12);
  EXPECT_EQ(layout.totalScalars, 19);
  EXPECT_TRUE(layout.hasNonOverlappingRanges());
}

TEST(RuntimeLayout, RuntimeSceneRefreshesBufferLayout)
{
  RuntimeScene scene;
  scene.dofs.appendRange(SubsystemId{0}, 6, 3);
  scene.geometry.appendPoint(SubsystemId{0}, 0, glm::dvec3{1.0, 2.0, 3.0});

  scene.refreshBufferLayout();

  EXPECT_EQ(scene.buffers.qScalars, 6);
  EXPECT_EQ(scene.buffers.qdotScalars, 6);
  EXPECT_EQ(scene.buffers.geometryPoints, 1);
}

}  // namespace
}  // namespace ksk::runtime
