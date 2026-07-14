#include <Scene/hair.h>
#include <Runtime/runtime-scene.h>

#include <algorithm>
#include <gtest/gtest.h>

namespace ksk::runtime {
namespace {

using HairObject = scene::HairObject;

struct RodObject {};

struct RodObjectDesc final : runtime::SceneObjectDesc {
  using ObjectType = RodObject;

  explicit RodObjectDesc(std::string tag = {})
      : runtime::SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] ObjectTypeId typeId() const noexcept override
  {
    return elementTypeId<RodObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "RodObject";
  }
};

TEST(RuntimeSceneDesc, RegistersAndFindsObjects)
{
  RuntimeSceneDesc scene;
  ObjectRef hair = scene.registerObject(
      std::make_unique<scene::HairObjectDesc>("hair"));
  ObjectRef rod = scene.registerObject(
      std::make_unique<RodObjectDesc>("der"));

  EXPECT_TRUE(hair.isValid());
  EXPECT_TRUE(rod.isValid());
  EXPECT_NE(hair.id, rod.id);
  EXPECT_TRUE(hair.isA<scene::HairObject>());
  EXPECT_FALSE(hair.isA<RodObject>());

  ObjectRef foundHair = scene.findObjectById(hair.id);
  EXPECT_TRUE(foundHair.isValid());
  EXPECT_EQ(foundHair.tag, "hair");
  EXPECT_TRUE(foundHair.isA<scene::HairObject>());

  const scene::HairObjectDesc* hairDesc =
      scene.findObjectDescAs<scene::HairObjectDesc>(foundHair);
  EXPECT_NE(hairDesc, nullptr);
  EXPECT_EQ(hairDesc->getTag(), "hair");

  ObjectRef foundRod = scene.findObjectById(rod.id);
  EXPECT_TRUE(foundRod.isValid());
  EXPECT_EQ(foundRod.tag, "der");
  EXPECT_TRUE(foundRod.isA<RodObject>());

  const RodObjectDesc* rodDesc =
      scene.findObjectDescAs<RodObjectDesc>(foundRod);
  EXPECT_NE(rodDesc, nullptr);
  EXPECT_EQ(rodDesc->getTag(), "der");

  EXPECT_FALSE(scene.findObjectById(-1).isValid());
}

TEST(RuntimeSceneDesc, QueriesObjectsByTypeAndTag)
{
  RuntimeSceneDesc scene;
  scene.registerObject(std::make_unique<scene::HairObjectDesc>("hair"));
  scene.registerObject(std::make_unique<scene::HairObjectDesc>("hair"));

  std::vector<ObjectRef> hairObjects = scene.findObjects<scene::HairObject>();
  EXPECT_EQ(hairObjects.size(), 2);
  EXPECT_TRUE(std::all_of(hairObjects.begin(), hairObjects.end(),
                          [](const ObjectRef& element) {
                            return element.isA<HairObject>();
                          }));

  scene.registerObject(std::make_unique<RodObjectDesc>("der"));
  std::vector<ObjectRef> derObjects = scene.findObjectsByTag("der");
  EXPECT_EQ(derObjects.size(), 1);
  EXPECT_EQ(derObjects[0].tag, "der");

  std::vector<ObjectRef> missingTag = scene.findObjectsByTag("missing");
  EXPECT_TRUE(missingTag.empty());
}

}  // namespace
}  // namespace ksk::runtime
