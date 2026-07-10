//
// test-device.cc
// Smoke test: create a Vulkan device. Skips if no compatible GPU.
//

#include <RHI/rhi.h>
#include <gtest/gtest.h>

using namespace ksk::rhi;

TEST(DeviceTest, CreateAndDestroy) {
  DeviceDesc desc{};
  desc.backend = Backend::Vulkan;
  desc.enableValidation = true;
  auto device = Device::create(desc);
  if (!device) {
    GTEST_SKIP() << "No Vulkan device available — skipping GPU test.";
  }
  EXPECT_EQ(device->backend(), Backend::Vulkan);
  EXPECT_FALSE(device->frameLoopActive());
}

TEST(DeviceTest, CreateMultipleDevicesIsAllowed) {
  // Plan §1.3: must support N devices simultaneously (no global GDynamicRHI).
  auto a = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!a) {
    GTEST_SKIP() << "No Vulkan device available — skipping GPU test.";
  }
  auto b = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  EXPECT_TRUE(a != nullptr);
  EXPECT_TRUE(b != nullptr);
}
