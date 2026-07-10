#pragma once
#include <chrono>
#include <string>
#include <spdlog/spdlog.h>

namespace ksk::core {

class Timer {
public:
  void start() { m_start = clock::now(); }

  double elapsedMs() const {
    auto now = clock::now();
    return std::chrono::duration<double, std::milli>(now - m_start).count();
  }

  double elapsedSec() const { return elapsedMs() / 1000.0; }

  void logElapsed(const std::string &label) const {
    spdlog::info("{}: {:.3f} ms", label, elapsedMs());
  }

private:
  using clock = std::chrono::high_resolution_clock;
  clock::time_point m_start{clock::now()};
};

} // namespace ksk::core
