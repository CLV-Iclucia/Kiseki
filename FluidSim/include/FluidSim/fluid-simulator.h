// ============================================================================
// include/FluidSim/fluid-simulator.h
// FluidSimulator: 薄编排器，构造即就绪
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <FluidSim/fluid-solver.h>
#include <Core/animation.h>
#include <memory>
#include <vector>

namespace fluid {

class FluidSimulator : public core::Animation {
public:
    /// 构造即就绪：传入 Context 和完整的 Solver 列表
    FluidSimulator(std::unique_ptr<FluidContext> ctx,
                   std::vector<std::shared_ptr<FluidModularSolver>> solvers);

    // ---- 运行 ----
    void step(core::Frame& frame) override;

    // ---- 访问 ----
    FluidContext& context() { return *ctx_; }
    const FluidContext& context() const { return *ctx_; }

    /// 按名称查找 Solver（用于运行时配置更新）
    FluidModularSolver* findSolver(std::string_view name);

    template<typename T>
    T* findSolver(std::string_view name) {
        return dynamic_cast<T*>(findSolver(name));
    }

    const std::vector<std::shared_ptr<FluidModularSolver>>& solvers() const {
        return solvers_;
    }

private:
    std::unique_ptr<FluidContext> ctx_;
    std::vector<std::shared_ptr<FluidModularSolver>> solvers_;

    Real computeCFL() const;
};

} // namespace fluid
