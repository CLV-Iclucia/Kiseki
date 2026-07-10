#pragma once
#include <Renderer/renderer.h>
#include <Renderer/scene-proxy.h>
#include <functional>
#include <thread>
#include <atomic>
#include <iostream>
#include <format>

namespace ksk::renderer {

/// SimulationApp: 将"模拟循环 + 渲染器 + 线程管理 + 帧推送"全部内聚。
///
/// 用户只需提供：
///   1. stepFn:       每步模拟的逻辑 (接受 step index)
///   2. buildProxy:   如何从当前状态生成渲染帧 (接受 step index)
///
/// 这样就**不可能**忘记推帧——框架自动在每步之后调用 buildProxy 并 push。
///
/// 用法：
///   SimulationApp app({.windowTitle = "My Sim"});
///   app.stepFn = [&](int step) { integrator->step(dt); };
///   app.buildProxy = [&](int step) { return buildSceneProxy(system, step); };
///   app.run(maxSteps);   // 阻塞直到模拟完成或窗口关闭
///
/// 如果 --no-render，可用 runHeadless() 直接跑模拟：
///   app.runHeadless(maxSteps);
///
struct SimulationApp {
  // ─── 用户必须设置的回调 ────────────────────────────────────────────────────

  /// 每步模拟执行的逻辑。step 从 0 开始递增。
  std::function<void(int step)> stepFn;

  /// 从当前系统状态构建渲染帧。每步之后自动调用。
  std::function<std::unique_ptr<SceneProxy>(int step)> buildProxy;

  // ─── 可选配置 ──────────────────────────────────────────────────────────────

  /// 渲染器配置
  RendererConfig rendererConfig;

  /// 每多少步打印一次日志 (0 = 不打印)
  int logInterval = 50;

  /// 自定义日志回调 (可选，不设置则用默认格式)
  std::function<void(int step)> logFn;

  /// 每步模拟后的可选回调 (可用于统计等，在 buildProxy 之后调用)
  std::function<void(int step)> onStepComplete;

  // ─── 构造 ──────────────────────────────────────────────────────────────────

  SimulationApp() = default;
  explicit SimulationApp(const RendererConfig& config) : rendererConfig(config) {}

  // ─── 运行接口 ──────────────────────────────────────────────────────────────

  /// 有渲染模式：开窗口 + 模拟线程 + 自动推帧。
  /// 阻塞直到模拟完成或窗口关闭。返回完成的步数。
  int run(int maxSteps) {
    if (!stepFn || !buildProxy) {
      std::cerr << "[SimulationApp] Error: stepFn and buildProxy must be set before run()\n";
      return 0;
    }

    auto renderer = createRenderer(rendererConfig);

    // 推送初始帧（step 0 之前的状态）
    renderer->queue().push(buildProxy(0));

    std::atomic<int> stepsCompleted{0};

    std::thread simThread([&]() {
      for (int step = 0; step < maxSteps && renderer->isRunning(); step++) {
        stepFn(step);

        // 自动推帧——用户不需要也不应该手动 push
        auto proxy = buildProxy(step + 1);
        renderer->queue().push(std::move(proxy));

        stepsCompleted = step + 1;

        if (logInterval > 0 && step % logInterval == 0) {
          if (logFn)
            logFn(step);
          else
            std::cout << std::format("[SimulationApp] Step {:4d} completed\n", step);
        }

        if (onStepComplete)
          onStepComplete(step);
      }
      renderer->shutdown();
    });

    // 渲染在主线程（满足 GLFW/macOS 要求）
    renderer->runOnCurrentThread();
    simThread.join();

    return stepsCompleted.load();
  }

  /// 无渲染模式：直接在当前线程跑模拟。返回完成的步数。
  int runHeadless(int maxSteps) {
    if (!stepFn) {
      std::cerr << "[SimulationApp] Error: stepFn must be set before runHeadless()\n";
      return 0;
    }

    for (int step = 0; step < maxSteps; step++) {
      stepFn(step);

      if (logInterval > 0 && step % logInterval == 0) {
        if (logFn)
          logFn(step);
        else
          std::cout << std::format("[SimulationApp] Step {:4d} completed\n", step);
      }

      if (onStepComplete)
        onStepComplete(step);
    }

    return maxSteps;
  }
};

} // namespace ksk::renderer
