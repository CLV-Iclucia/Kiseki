// ============================================================================
// Core/include/Core/error.h
// 统一错误报告：日志 + 抛异常（C++ 端有 spdlog 记录，Python 端 pybind11 自动转换）
// ============================================================================
#pragma once

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace sim::core {

/// 带源码位置的致命错误：记录日志并抛出 std::runtime_error。
/// pybind11 会自动把 std::runtime_error 映射为 Python RuntimeError。
[[noreturn]] inline void throwError(const char* file, int line, const std::string& msg) {
    std::string full = std::string(file) + ":" + std::to_string(line) + ": " + msg;
    spdlog::error(full);
    throw std::runtime_error(full);
}

} // namespace sim::core

/// 报错并抛异常。C++ 端有日志，Python 端 pybind11 自动捕获为 RuntimeError。
#define SIM_THROW(msg) ::sim::core::throwError(__FILE__, __LINE__, (msg))

/// 条件断言：condition 不满足时报错抛异常。
#define SIM_ASSERT(condition, msg) \
    do { if (!(condition)) SIM_THROW(msg); } while(0)
