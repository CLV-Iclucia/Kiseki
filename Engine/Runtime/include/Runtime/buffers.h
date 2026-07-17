#pragma once

#include <RHI/buffer.h>
#include <RHI/buffer-utils.h>

#include <Eigen/Core>
#include <glm/glm.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Runtime/dof-layout.h>

namespace ksk::rhi {

class Device;

}  // namespace ksk::rhi

namespace ksk::runtime {

class ConstDofView {
 public:
  ConstDofView() = default;

  [[nodiscard]] static ConstDofView CPU(const double* values, int count)
  {
    return CPU(values, 0, count);
  }

  [[nodiscard]] static ConstDofView CPU(const double* values,
                                        int scalarOffset,
                                        int scalarCount)
  {
    ConstDofView view;
    view.storage_ = values;
    view.offset_ = scalarOffset;
    view.count_ = scalarCount;
    return view;
  }

  [[nodiscard]] static ConstDofView GPU(rhi::Device& device,
                                        rhi::BufferRef buffer,
                                        int scalarOffset,
                                        int scalarCount)
  {
    ConstDofView view;
    view.device_ = &device;
    view.storage_ = std::move(buffer);
    view.offset_ = scalarOffset;
    view.count_ = scalarCount;
    return view;
  }

  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }
  [[nodiscard]] int scalarOffset() const noexcept { return offset_; }
  [[nodiscard]] int scalarCount() const noexcept { return count_; }

  [[nodiscard]] std::span<const double> cpu() const
  {
    assert(isCPU());
    return std::span<const double>(
        std::get<const double*>(storage_), static_cast<size_t>(count_));
  }

  [[nodiscard]] const double& operator[](int index) const
  {
    assert(isCPU());
    assert(index >= 0 && index < count_);
    return std::get<const double*>(storage_)[index];
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

 protected:
  rhi::Device* device_ = nullptr;
  std::variant<const double*, rhi::BufferRef> storage_ =
      static_cast<const double*>(nullptr);
  int offset_ = 0;
  int count_ = 0;
};

class DofView {
 public:
  DofView() = default;

  [[nodiscard]] static DofView CPU(double* values, int count)
  {
    return CPU(values, 0, count);
  }

  [[nodiscard]] static DofView CPU(double* values,
                                   int scalarOffset,
                                   int scalarCount)
  {
    DofView view;
    view.storage_ = values;
    view.offset_ = scalarOffset;
    view.count_ = scalarCount;
    return view;
  }

  [[nodiscard]] static DofView GPU(rhi::Device& device,
                                   rhi::BufferRef buffer,
                                   int scalarOffset,
                                   int scalarCount)
  {
    DofView view;
    view.device_ = &device;
    view.storage_ = std::move(buffer);
    view.offset_ = scalarOffset;
    view.count_ = scalarCount;
    return view;
  }

  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }
  [[nodiscard]] int scalarOffset() const noexcept { return offset_; }
  [[nodiscard]] int scalarCount() const noexcept { return count_; }

  [[nodiscard]] std::span<double> cpu()
  {
    assert(isCPU());
    return std::span<double>(
        std::get<double*>(storage_), static_cast<size_t>(count_));
  }

  [[nodiscard]] double& operator[](int index)
  {
    assert(isCPU());
    assert(index >= 0 && index < count_);
    return std::get<double*>(storage_)[index];
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

  [[nodiscard]] ConstDofView asConst() const
  {
    if (isCPU()) {
      return ConstDofView::CPU(std::get<double*>(storage_), offset_, count_);
    }
    return ConstDofView::GPU(*device_, std::get<rhi::BufferRef>(storage_),
                             offset_, count_);
  }

  [[nodiscard]] operator ConstDofView() const { return asConst(); }

 private:
  rhi::Device* device_ = nullptr;
  std::variant<double*, rhi::BufferRef> storage_ = static_cast<double*>(nullptr);
  int offset_ = 0;
  int count_ = 0;
};

class ConstGeometryView {
 public:
  ConstGeometryView() = default;

  [[nodiscard]] static ConstGeometryView CPU(const glm::dvec3* values,
                                             int count)
  {
    return CPU(values, 0, count);
  }

  [[nodiscard]] static ConstGeometryView CPU(const glm::dvec3* values,
                                             int pointOffset,
                                             int pointCount)
  {
    ConstGeometryView view;
    view.storage_ = values;
    view.offset_ = pointOffset;
    view.count_ = pointCount;
    return view;
  }

  [[nodiscard]] static ConstGeometryView GPU(rhi::Device& device,
                                             rhi::BufferRef buffer,
                                             int pointOffset,
                                             int pointCount)
  {
    ConstGeometryView view;
    view.device_ = &device;
    view.storage_ = std::move(buffer);
    view.offset_ = pointOffset;
    view.count_ = pointCount;
    return view;
  }

  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }
  [[nodiscard]] int pointOffset() const noexcept { return offset_; }
  [[nodiscard]] int pointCount() const noexcept { return count_; }

  [[nodiscard]] std::span<const glm::dvec3> cpu() const
  {
    assert(isCPU());
    return std::span<const glm::dvec3>(
        std::get<const glm::dvec3*>(storage_), static_cast<size_t>(count_));
  }

  [[nodiscard]] const glm::dvec3& operator[](int index) const
  {
    assert(isCPU());
    assert(index >= 0 && index < count_);
    return std::get<const glm::dvec3*>(storage_)[index];
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

 protected:
  rhi::Device* device_ = nullptr;
  std::variant<const glm::dvec3*, rhi::BufferRef> storage_ =
      static_cast<const glm::dvec3*>(nullptr);
  int offset_ = 0;
  int count_ = 0;
};

class GeometryView {
 public:
  GeometryView() = default;

  [[nodiscard]] static GeometryView CPU(glm::dvec3* values, int count)
  {
    return CPU(values, 0, count);
  }

  [[nodiscard]] static GeometryView CPU(glm::dvec3* values,
                                        int pointOffset,
                                        int pointCount)
  {
    GeometryView view;
    view.storage_ = values;
    view.offset_ = pointOffset;
    view.count_ = pointCount;
    return view;
  }

  [[nodiscard]] static GeometryView GPU(rhi::Device& device,
                                        rhi::BufferRef buffer,
                                        int pointOffset,
                                        int pointCount)
  {
    GeometryView view;
    view.device_ = &device;
    view.storage_ = std::move(buffer);
    view.offset_ = pointOffset;
    view.count_ = pointCount;
    return view;
  }

  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] bool isCPU() const noexcept { return device_ == nullptr; }
  [[nodiscard]] bool isGPU() const noexcept { return device_ != nullptr; }
  [[nodiscard]] rhi::Device* device() const noexcept { return device_; }
  [[nodiscard]] int pointOffset() const noexcept { return offset_; }
  [[nodiscard]] int pointCount() const noexcept { return count_; }

  [[nodiscard]] std::span<glm::dvec3> cpu()
  {
    assert(isCPU());
    return std::span<glm::dvec3>(
        std::get<glm::dvec3*>(storage_), static_cast<size_t>(count_));
  }

  [[nodiscard]] glm::dvec3& operator[](int index)
  {
    assert(isCPU());
    assert(index >= 0 && index < count_);
    return std::get<glm::dvec3*>(storage_)[index];
  }

  [[nodiscard]] rhi::BufferRef gpu() const
  {
    assert(isGPU());
    return std::get<rhi::BufferRef>(storage_);
  }

  [[nodiscard]] ConstGeometryView asConst() const
  {
    if (isCPU()) {
      return ConstGeometryView::CPU(std::get<glm::dvec3*>(storage_), offset_,
                                    count_);
    }
    return ConstGeometryView::GPU(*device_, std::get<rhi::BufferRef>(storage_),
                                  offset_, count_);
  }

  [[nodiscard]] operator ConstGeometryView() const { return asConst(); }

 private:
  rhi::Device* device_ = nullptr;
  std::variant<glm::dvec3*, rhi::BufferRef> storage_ =
      static_cast<glm::dvec3*>(nullptr);
  int offset_ = 0;
  int count_ = 0;
};

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

  [[nodiscard]] static DofBuffer GPU(rhi::Device& device, int scalarCount)
  {
    return FromGPU(device,
                   rhi::createDeviceLocalBuffer(
                       device,
                       sizeof(double) * static_cast<size_t>(scalarCount),
                       "runtime-dof-buffer"));
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

  [[nodiscard]] bool contains(const DofRange& range) const noexcept
  {
    const int total = scalarCount();
    return range.scalarOffset >= 0 && range.scalarCount >= 0 &&
           range.scalarOffset <= total &&
           range.scalarCount <= total - range.scalarOffset;
  }

  [[nodiscard]] DofView view()
  {
    return slice(0, scalarCount());
  }

  [[nodiscard]] ConstDofView view() const
  {
    return slice(0, scalarCount());
  }

  [[nodiscard]] DofView slice(const DofRange& range)
  {
    return slice(range.scalarOffset, range.scalarCount);
  }

  [[nodiscard]] ConstDofView slice(const DofRange& range) const
  {
    return slice(range.scalarOffset, range.scalarCount);
  }

  [[nodiscard]] DofView slice(int scalarOffset, int scalarCount)
  {
    requireRange(scalarOffset, scalarCount, this->scalarCount(),
                 "DofBuffer::slice");
    if (isCPU()) {
      return DofView::CPU(cpu().data() + scalarOffset, scalarOffset,
                          scalarCount);
    }
    return DofView::GPU(*device_, gpu(), scalarOffset, scalarCount);
  }

  [[nodiscard]] ConstDofView slice(int scalarOffset, int scalarCount) const
  {
    requireRange(scalarOffset, scalarCount, this->scalarCount(),
                 "DofBuffer::slice");
    if (isCPU()) {
      return ConstDofView::CPU(cpu().data() + scalarOffset, scalarOffset,
                               scalarCount);
    }
    return ConstDofView::GPU(*device_, gpu(), scalarOffset, scalarCount);
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
  static void requireRange(int offset,
                           int count,
                           int total,
                           const char* operation)
  {
    if (offset < 0 || count < 0 || offset > total || count > total - offset) {
      throw std::out_of_range(std::string(operation) +
                              " received a range outside the buffer");
    }
  }

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

  [[nodiscard]] bool contains(int pointOffset, int pointCount) const noexcept
  {
    const int total = this->pointCount();
    return pointOffset >= 0 && pointCount >= 0 && pointOffset <= total &&
           pointCount <= total - pointOffset;
  }

  [[nodiscard]] GeometryView view()
  {
    return slice(0, pointCount());
  }

  [[nodiscard]] ConstGeometryView view() const
  {
    return slice(0, pointCount());
  }

  [[nodiscard]] GeometryView slice(int pointOffset, int pointCount)
  {
    requireRange(pointOffset, pointCount, this->pointCount(),
                 "GeometryBuffer::slice");
    if (isCPU()) {
      return GeometryView::CPU(cpu().data() + pointOffset, pointOffset,
                               pointCount);
    }
    return GeometryView::GPU(*device_, gpu(), pointOffset, pointCount);
  }

  [[nodiscard]] ConstGeometryView slice(int pointOffset, int pointCount) const
  {
    requireRange(pointOffset, pointCount, this->pointCount(),
                 "GeometryBuffer::slice");
    if (isCPU()) {
      return ConstGeometryView::CPU(cpu().data() + pointOffset, pointOffset,
                                    pointCount);
    }
    return ConstGeometryView::GPU(*device_, gpu(), pointOffset, pointCount);
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
  static void requireRange(int offset,
                           int count,
                           int total,
                           const char* operation)
  {
    if (offset < 0 || count < 0 || offset > total || count > total - offset) {
      throw std::out_of_range(std::string(operation) +
                              " received a range outside the buffer");
    }
  }

  rhi::Device* device_ = nullptr;
  std::variant<std::vector<glm::dvec3>, rhi::BufferRef> storage_;
};

}  // namespace ksk::runtime
