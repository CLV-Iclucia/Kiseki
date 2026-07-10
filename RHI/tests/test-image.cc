//
// test-image.cc
// Create a small 3D image, clear to a known color, read back via
// copy_image_to_buffer, verify pixel contents.
//

#include <RHI/rhi.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace ksk::rhi;

TEST(ImageTest, ClearAndReadback3D) {
  auto device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  // Tiny 4x4x4 RGBA8 storage image.
  constexpr uint32_t W = 4, H = 4, D = 4;
  ImageDesc idesc{};
  idesc.dim = ImageDesc::Dim::D3;
  idesc.width = W;
  idesc.height = H;
  idesc.depth = D;
  idesc.format = Format::RGBA8_UNorm;
  idesc.usage = ImageDesc::Storage | ImageDesc::TransferDst | ImageDesc::TransferSrc;
  idesc.debugName = "test-3d";
  auto image = device->createImage(idesc);
  ASSERT_TRUE(image.valid());
  EXPECT_EQ(image->desc().width, W);

  // Readback buffer: WxHxD * 4 bytes.
  const size_t bytes = static_cast<size_t>(W) * H * D * 4;
  auto readback = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
      .debugName = "image-readback",
  });

  auto cmd = device->beginCommands(QueueType::Graphics);

  // Transition Undefined -> TransferDst.
  {
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageTransfer;
    b.dstStage = BarrierDesc::StageTransfer;
    b.srcAccess = 0;
    b.dstAccess = BarrierDesc::AccessTransferWrite;
    BarrierDesc::ImageBarrier ib{};
    ib.image = image;
    ib.oldLayout = BarrierDesc::ImageBarrier::Layout::Undefined;
    ib.newLayout = BarrierDesc::ImageBarrier::Layout::TransferDst;
    b.imageBarriers.push_back(ib);
    cmd->barrier(b);
  }

  // Clear to (255, 128, 64, 255).
  ClearValue c = ClearValue::makeColorF(1.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f);
  cmd->clearImage(image, c);

  // Transition TransferDst -> TransferSrc.
  {
    BarrierDesc b{};
    b.srcStage = BarrierDesc::StageTransfer;
    b.dstStage = BarrierDesc::StageTransfer;
    b.srcAccess = BarrierDesc::AccessTransferWrite;
    b.dstAccess = BarrierDesc::AccessTransferRead;
    BarrierDesc::ImageBarrier ib{};
    ib.image = image;
    ib.oldLayout = BarrierDesc::ImageBarrier::Layout::TransferDst;
    ib.newLayout = BarrierDesc::ImageBarrier::Layout::TransferSrc;
    b.imageBarriers.push_back(ib);
    cmd->barrier(b);
  }

  // Copy whole image to buffer.
  std::array<BufferImageCopy, 1> rs{};
  rs[0].imageExtentW = W;
  rs[0].imageExtentH = H;
  rs[0].imageExtentD = D;
  rs[0].layerCount = 1;
  cmd->copyImageToBuffer(image, readback, rs);

  device->submitAndWait(*cmd, QueueType::Graphics);

  // Verify pixel contents.
  auto pixels = readback->mapTyped<uint8_t>();
  ASSERT_EQ(pixels.size(), bytes);
  // Expected RGBA8 values from the clear color.
  uint8_t expR = 255, expG = 128, expB = 64, expA = 255;
  bool ok = true;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    if (pixels[i + 0] != expR || pixels[i + 1] != expG || pixels[i + 2] != expB ||
        pixels[i + 3] != expA) {
      ok = false;
      break;
    }
  }
  EXPECT_TRUE(ok);
  readback->unmap();
}
