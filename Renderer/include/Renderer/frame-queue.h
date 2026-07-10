#pragma once
#include <Renderer/scene-proxy.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace ksk::renderer {

class FrameQueue {
public:
  explicit FrameQueue(size_t maxCapacity = 16) : m_maxCapacity(maxCapacity) {}

  void push(std::unique_ptr<SceneProxy> frame) {
    std::unique_lock lock(m_mutex);
    m_cvProducer.wait(lock, [&] { return m_queue.size() < m_maxCapacity || m_shutdown; });
    if (m_shutdown) return;
    m_queue.push(std::move(frame));
    m_cvConsumer.notify_one();
  }

  /// 返回 nullptr 表示已 shutdown。
  std::unique_ptr<SceneProxy> pop() {
    std::unique_lock lock(m_mutex);
    m_cvConsumer.wait(lock, [&] { return !m_queue.empty() || m_shutdown; });
    if (m_shutdown && m_queue.empty()) return nullptr;
    auto frame = std::move(m_queue.front());
    m_queue.pop();
    m_cvProducer.notify_one();
    return frame;
  }

  std::unique_ptr<SceneProxy> tryPop() {
    std::lock_guard lock(m_mutex);
    if (m_queue.empty()) return nullptr;
    auto frame = std::move(m_queue.front());
    m_queue.pop();
    m_cvProducer.notify_one();
    return frame;
  }

  void shutdown() {
    std::lock_guard lock(m_mutex);
    m_shutdown = true;
    m_cvConsumer.notify_all();
    m_cvProducer.notify_all();
  }

  [[nodiscard]] size_t size() const {
    std::lock_guard lock(m_mutex);
    return m_queue.size();
  }

  [[nodiscard]] bool isShutdown() const {
    std::lock_guard lock(m_mutex);
    return m_shutdown;
  }

private:
  mutable std::mutex m_mutex;
  std::condition_variable m_cvConsumer;
  std::condition_variable m_cvProducer;
  std::queue<std::unique_ptr<SceneProxy>> m_queue;
  size_t m_maxCapacity;
  bool m_shutdown = false;
};

} // namespace ksk::renderer
