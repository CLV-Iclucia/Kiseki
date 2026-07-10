//
// buffer-utils.h
// Common buffer creation helpers.
//

#pragma once

#include <RHI/device.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace ksk::rhi {

// Creates a device-local buffer suitable for compute workloads and staging
// transfers.
inline BufferRef createDeviceLocalBuffer(
    Device& device,
    size_t sizeBytes,
    std::string_view debugName = {})
{
    return device.createBuffer({
        .sizeBytes = std::max<size_t>(sizeBytes, 4),
        .visibility = BufferDesc::Visibility::DeviceLocal,
        .usage = BufferDesc::Storage |
                 BufferDesc::TransferSrc |
                 BufferDesc::TransferDst,
        .debugName = std::string(debugName),
    });
}

} // namespace ksk::rhi
