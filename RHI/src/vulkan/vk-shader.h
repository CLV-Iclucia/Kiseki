//
// vk-shader.h
// Vulkan implementation of Shader.
// See docs/rhi-plan.md §3.4.2.
//

#pragma once

#include <RHI/reflection.h>
#include <RHI/shader.h>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace ksk::rhi::vulkan {

class VulkanDevice;

class VulkanShader final : public Shader {
 public:
  // Constructor reflects bytecode into ReflectionInfo and creates the
  // VkShaderModule. Throws std::runtime_error on backend failure.
  VulkanShader(VulkanDevice* device, std::span<const std::byte> bytecode,
               ShaderStage stage, std::string_view entryPoint);
  ~VulkanShader() override;

  // ---- Shader interface ---------------------------------------------------
  ShaderStage stage() const override { return m_stage; }
  std::string_view entryPoint() const override { return m_entryPoint; }
  const ReflectionInfo& reflection() const override { return m_reflection; }

  // ---- Backend-internal ---------------------------------------------------
  VkShaderModule vkHandle() const noexcept { return m_module; }

 protected:
  void destroy() noexcept override;

 private:
  VulkanDevice* m_device;
  ShaderStage m_stage;
  std::string m_entryPoint;
  VkShaderModule m_module = VK_NULL_HANDLE;
  ReflectionInfo m_reflection;
};

}  // namespace ksk::rhi::vulkan
