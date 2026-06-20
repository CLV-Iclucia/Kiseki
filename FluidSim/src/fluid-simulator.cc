// ============================================================================
// src/fluid-simulator.cc
// FluidSimulator 编排器实现
// ============================================================================

#include <FluidSim/fluid-simulator.h>
#include <algorithm>
#include <format>
#include <iostream>

namespace fluid {

FluidSimulator::FluidSimulator(
    std::unique_ptr<FluidContext> ctx,
    std::vector<std::shared_ptr<FluidModularSolver>> solvers)
    : ctx_(std::move(ctx))
    , solvers_(std::move(solvers))
{
    // 构造完成即就绪，无额外初始化步骤
}

void FluidSimulator::step(core::Frame& frame) {
    Real dt = frame.dt;
    Real t = 0;
    int substepCount = 0;

    std::cout << std::format("********* Frame {} *********", frame.idx)
              << std::endl;

    ctx_->beginFrame();  // GPU: begin command list; CPU: no-op

    while (t < dt) {
        Real subDt = std::min(computeCFL(), dt - t);
        substepCount++;
        std::cout << std::format("<<<<< Substep {}, dt = {} >>>>>",
                                 substepCount, subDt) << std::endl;
        for (auto& solver : solvers_) {
            solver->solve(*ctx_, subDt);
        }
        t += subDt;
    }

    ctx_->endFrame();    // GPU: submit + wait; CPU: no-op
    frame.onAdvance();
}

FluidModularSolver* FluidSimulator::findSolver(std::string_view name) {
    for (auto& s : solvers_)
        if (s->name() == name) return s.get();
    return nullptr;
}

Real FluidSimulator::computeCFL() const {
    // 由 Context 中的速度场计算 CFL。
    // 基类无法直接访问速度场（在子类中），因此提供一个保守的默认值。
    // 具体的 CFL 计算由预设工厂在构建管线时通过插入一个 CFL-aware 的
    // 专用 Solver 或由 Context 子类实现。这里暂时返回一个大值让子步不做分割。
    // 实际实现中，FreeSurface 预设会覆写此逻辑。
    return ctx_->gridSpacing * 5.0;
}

} // namespace fluid
