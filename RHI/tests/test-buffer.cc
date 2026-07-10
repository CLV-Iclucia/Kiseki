//
// test-buffer.cc
// HostVisible roundtrip + DeviceLocal staging upload/readback.
//

#include <RHI/rhi.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <numeric>

using namespace ksk::rhi;

TEST(BufferTest, HostVisibleRoundtrip) {
  auto device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  constexpr size_t N = 256;
  BufferDesc desc{};
  desc.sizeBytes = N * sizeof(uint32_t);
  desc.visibility = BufferDesc::Visibility::HostVisible;
  desc.usage = BufferDesc::TransferSrc | BufferDesc::TransferDst;
  desc.debugName = "host-buf";
  auto buf = device->createBuffer(desc);
  ASSERT_TRUE(buf.valid());
  EXPECT_EQ(buf->sizeBytes(), N * sizeof(uint32_t));

  // Write
  auto span = buf->mapTyped<uint32_t>();
  ASSERT_EQ(span.size(), N);
  for (size_t i = 0; i < N; ++i) span[i] = static_cast<uint32_t>(i * 7 + 1);
  buf->unmap();

  // Read
  auto span2 = buf->mapTyped<uint32_t>();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(span2[i], i * 7 + 1);
  }
  buf->unmap();
}

TEST(BufferTest, DeviceLocalViaStaging) {
  auto device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  constexpr size_t N = 128;
  const size_t bytes = N * sizeof(uint32_t);

  // Source: HostVisible
  auto staging = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::HostVisible,
      .usage = BufferDesc::TransferSrc,
      .debugName = "staging",
  });
  // Dest: DeviceLocal
  auto gpu = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::DeviceLocal,
      .usage = BufferDesc::TransferSrc | BufferDesc::TransferDst,
      .debugName = "gpu-buf",
  });
  // Readback: HostVisible (or Readback)
  auto readback = device->createBuffer({
      .sizeBytes = bytes,
      .visibility = BufferDesc::Visibility::Readback,
      .usage = BufferDesc::TransferDst,
      .debugName = "readback",
  });

  // Fill staging.
  auto src = staging->mapTyped<uint32_t>();
  for (size_t i = 0; i < N; ++i) src[i] = static_cast<uint32_t>(i + 100);
  staging->unmap();

  // Record: staging -> gpu, then gpu -> readback.
  auto cmd = device->beginCommands(QueueType::Transfer);
  std::array<BufferCopy, 1> region{{{0, 0, bytes}}};
  cmd->copyBuffer(staging, gpu, region);

  // Insert a barrier between the two copies.
  BarrierDesc b{};
  b.srcStage = BarrierDesc::StageTransfer;
  b.dstStage = BarrierDesc::StageTransfer;
  b.srcAccess = BarrierDesc::AccessTransferWrite;
  b.dstAccess = BarrierDesc::AccessTransferRead;
  cmd->barrier(b);

  cmd->copyBuffer(gpu, readback, region);
  device->submitAndWait(*cmd, QueueType::Transfer);

  // Verify.
  auto dst = readback->mapTyped<uint32_t>();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(dst[i], i + 100) << "mismatch at " << i;
  }
  readback->unmap();
}

TEST(BufferTest, CreateDeviceLocalBufferPreset) {
  auto device = Device::create({.backend = Backend::Vulkan, .enableValidation = true});
  if (!device) GTEST_SKIP() << "No Vulkan device.";

  constexpr size_t bytes = 64 * sizeof(uint32_t);
  auto buffer = createDeviceLocalBuffer(*device, bytes, "device-local-preset");

  ASSERT_TRUE(buffer.valid());
  EXPECT_EQ(buffer->sizeBytes(), bytes);
}
