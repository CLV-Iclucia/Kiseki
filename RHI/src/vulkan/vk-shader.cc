//
// vk-shader.cc
//

#include "vk-shader.h"

#include "vk-device.h"
#include "vk-internals.h"

namespace ksk::rhi::vulkan {

VulkanShader::VulkanShader(VulkanDevice* device,
                           std::span<const std::byte> bytecode,
                           ShaderStage stage, std::string_view entryPoint)
    : m_device(device), m_stage(stage), m_entryPoint(entryPoint) {
  // SPIR-V is 4-byte aligned, native uint32_t units. The vendor's
  // VkShaderModuleCreateInfo accepts a byte size + uint32_t* — alignment is
  // expected because SPIR-V module is 4-byte aligned by spec.
  if (bytecode.size_bytes() % 4 != 0) {
    spdlog::error(
        "[VulkanShader] SPIR-V bytecode size ({} bytes) is not a multiple of "
        "4 — module is malformed",
        bytecode.size_bytes());
    throw std::runtime_error("malformed SPIR-V bytecode (non-4-byte size)");
  }

  // Reflect once; surface to Shader::reflection().
  m_reflection = reflectSpirv(bytecode, stage, entryPoint);
  if (m_reflection.bindings.empty() && m_reflection.pushConstants.empty()) {
    // Reflection emits its own log at error level — no extra noise here. A
    // truly empty kernel is theoretically valid (just a numthreads wrapper),
    // so this is not a hard failure.
    spdlog::debug(
        "[VulkanShader] reflection produced 0 bindings / 0 push constants for "
        "entry '{}'; kernel may be intentionally empty",
        m_entryPoint);
  }

  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytecode.size_bytes();
  ci.pCode = reinterpret_cast<const uint32_t*>(bytecode.data());
  VK_CHECK(vkCreateShaderModule(m_device->vkDevice(), &ci, nullptr, &m_module));
}

VulkanShader::~VulkanShader() = default;

void VulkanShader::destroy() noexcept {
  // Tier 0: drain the device before destroying. Plan §3.0 promises the
  // Tier 1 deferred-destroy path lands at R8.
  if (m_module != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device->vkDevice());
    vkDestroyShaderModule(m_device->vkDevice(), m_module, nullptr);
    m_module = VK_NULL_HANDLE;
  }
  delete this;
}

}  // namespace ksk::rhi::vulkan
