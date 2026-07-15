#pragma once

#include <RHI/buffer.h>
#include <RHI/buffer-utils.h>

#include <Eigen/Core>
#include <glm/glm.hpp>

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ksk::rhi {

class Device;

}  // namespace ksk::rhi

namespace ksk::runtime {

class DofBuffer {
 public:
  DofBuffer() = default;

  [[nodiscard]] static DofBuffer FromCPU(Eigen::VectorXd values)
  {
    DofBuffer buffer;
    buffer.device_ = nullptr;
    buffer.storage_ = std::move(values);
    return buffer;
  }

  [[nodiscard]] static DofBuffer CPU(int scalarCount)
  {
    return FromCPU(Eigen::VectorXd::Zero(scalarCount));
  }

  [[nodiscard]] static DofBuffer FromGPU(rhi::Device& device,
                                         rhi::BufferRef buffer)
  {
    DofBuffer result;
    result.device_ = &device;
    result.storage_ = std::move(buffer);
    return result;
  }

  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }

  [[nodiscard]] int scalarCount() const
  {
    if (isCPU()) {
      return static_cast<int>(cpu().size());
    }
    return static_cast<int>(gpu()->sizeBytes() / sizeof(double));
  }

  [[nodiscard]] DofBuffer clone() const
  {
    if (isCPU()) {
      return FromCPU(cpu());
    }
    throw std::runtime_error("GPU DofBuffer clone is not implemented");
  }

  void setZero()
  {
    if (isCPU()) {
      cpu().setZero();
      return;
    }
    throw std::runtime_error("GPU DofBuffer setZero is not implemented");
  }

  void copyFrom(const DofBuffer& other)
  {
    if (isCPU() && other.isCPU()) {
      storage_ = other.cpu();
      return;
    }
    requireCompatible(other, "DofBuffer::copyFrom");
    throw std::runtime_error("GPU DofBuffer copyFrom is not implemented");
  }

  void assignScaled(double scale, const DofBuffer& input)
  {
    requireCompatible(input, "DofBuffer::assignScaled");
    if (isCPU()) {
      cpu() = (scale * input.cpu()).eval();
      return;
    }
    throw std::runtime_error("GPU DofBuffer assignScaled is not implemented");
  }

  void assignLinearCombination(const DofBuffer& base,
                               double scale,
                               const DofBuffer& delta)
  {
    requireCompatible(base, "DofBuffer::assignLinearCombination");
    requireCompatible(delta, "DofBuffer::assignLinearCombination");
    if (isCPU()) {
      cpu() = (base.cpu() + scale * delta.cpu()).eval();
      return;
    }
    throw std::runtime_error(
        "GPU DofBuffer assignLinearCombination is not implemented");
  }

  void addScaled(double scale, const DofBuffer& input)
  {
    requireCompatible(input, "DofBuffer::addScaled");
    if (isCPU()) {
      cpu() += scale * input.cpu();
      return;
    }
    throw std::runtime_error("GPU DofBuffer addScaled is not implemented");
  }

  [[nodiscard]] double dot(const DofBuffer& other) const
  {
    requireCompatible(other, "DofBuffer::dot");
    if (isCPU()) {
      return cpu().dot(other.cpu());
    }
    throw std::runtime_error("GPU DofBuffer dot is not implemented");
  }

  [[nodiscard]] double norm() const
  {
    if (isCPU()) {
      return cpu().norm();
    }
    throw std::runtime_error("GPU DofBuffer norm is not implemented");
  }

  [[nodiscard]] Eigen::VectorXd& cpu()
  {
    assert(isCPU());
    return std::get<Eigen::VectorXd>(storage_);
  }

  [[nodiscard]] const Eigen::VectorXd& cpu() const
  {
    assert(isCPU());
    return std::get<Eigen::VectorXd>(storage_);
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

 private:
  void requireCompatible(const DofBuffer& other, const char* operation) const
  {
    if (isCPU() != other.isCPU()) {
      throw std::runtime_error(std::string(operation) +
                               " received buffers on different memory spaces");
    }
    if (device_ != other.device_) {
      throw std::runtime_error(std::string(operation) +
                               " received buffers on different devices");
    }
    if (scalarCount() != other.scalarCount()) {
      throw std::runtime_error(std::string(operation) +
                               " received buffers with different sizes");
    }
  }

  rhi::Device* device_ = nullptr;
  std::variant<Eigen::VectorXd, rhi::BufferRef> storage_;
};

class GeometryBuffer {
 public:
  GeometryBuffer() = default;

  [[nodiscard]] static GeometryBuffer FromCPU(std::vector<glm::dvec3> values)
  {
    GeometryBuffer buffer;
    buffer.device_ = nullptr;
    buffer.storage_ = std::move(values);
    return buffer;
  }

  [[nodiscard]] static GeometryBuffer CPU(int pointCount)
  {
    return FromCPU(std::vector<glm::dvec3>(
        static_cast<size_t>(pointCount), glm::dvec3(0.0)));
  }

  [[nodiscard]] static GeometryBuffer GPU(rhi::Device& device, int pointCount)
  {
    return FromGPU(device,
                   rhi::createDeviceLocalBuffer(
                       device,
                       sizeof(glm::dvec3) * static_cast<size_t>(pointCount),
                       "runtime-geometry-buffer"));
  }

  [[nodiscard]] static GeometryBuffer FromGPU(rhi::Device& device,
                                              rhi::BufferRef buffer)
  {
    GeometryBuffer result;
    result.device_ = &device;
    result.storage_ = std::move(buffer);
    return result;
  }

  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }

  [[nodiscard]] int pointCount() const
  {
    if (isCPU()) {
      return static_cast<int>(cpu().size());
    }
    return static_cast<int>(gpu()->sizeBytes() / sizeof(glm::dvec3));
  }

  [[nodiscard]] std::vector<glm::dvec3>& cpu()
  {
    assert(isCPU());
    return std::get<std::vector<glm::dvec3>>(storage_);
  }

  [[nodiscard]] const std::vector<glm::dvec3>& cpu() const
  {
    assert(isCPU());
    return std::get<std::vector<glm::dvec3>>(storage_);
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

 private:
  rhi::Device* device_ = nullptr;
  std::variant<std::vector<glm::dvec3>, rhi::BufferRef> storage_;
};

}  // namespace ksk::runtime
