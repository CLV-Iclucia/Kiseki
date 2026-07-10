//
// test-fence.cc
// Submit an empty barrier-only command list signaling a fence; verify wait.
//

#include <RHI/rhi.h>
#include <gtest/gtest.h>

using namespace ksk::rhi;

TEST(FenceTest, SignalAfterSubmit) {
  auto device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  auto fence = device->createFence();
  EXPECT_FALSE(device->isFenceSignaled(*fence));

  auto cmd = device->beginCommands(QueueType::Graphics);
  // Issue a no-op global barrier so the cmd list isn't empty.
  BarrierDesc b{};
  b.srcStage = BarrierDesc::StageTransfer;
  b.dstStage = BarrierDesc::StageTransfer;
  b.srcAccess = BarrierDesc::AccessTransferWrite;
  b.dstAccess = BarrierDesc::AccessTransferRead;
  cmd->barrier(b);

  device->submit(*cmd, {}, {}, fence.get());
  device->waitFence(*fence);
  EXPECT_TRUE(device->isFenceSignaled(*fence));
}
