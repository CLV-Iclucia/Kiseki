// ============================================================================
// include/FluidSim/fluid-solver.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-context.h>
#include <Core/properties.h>
#include <string>
#include <string_view>

namespace fluid {

// ---- 通用基类（Python binding 和 Simulator 持有的类型）----

class FluidModularSolver : NonCopyable {
public:
    explicit FluidModularSolver(std::string name) : name_(std::move(name)) {}
    virtual ~FluidModularSolver() = default;

    /// 执行一个子步
    virtual void solve(FluidContext& ctx, Real dt) = 0;

    /// 运行时配置更新（类型擦除，Python/UI 友好）
    virtual void configure(const core::Properties& props) { (void)props; }

    /// 查询当前配置
    virtual core::Properties currentConfig() const { return {}; }

    /// Solver 名称（调试/查找用）
    std::string_view name() const { return name_; }

private:
    std::string name_;
};

} // namespace fluid
