// ============================================================================
// include/FluidSim/cpu/cpu-solver.h
// CPUModularSolver: CPU 后端的中间层基类
// 将 FluidContext& 转换为 CPUFluidContext&，消除每个 Solver 中的重复 cast。
// ============================================================================
#pragma once

#include <FluidSim/fluid-solver.h>
#include <FluidSim/cpu/cpu-context.h>

namespace fluid::cpu {

class CPUModularSolver : public FluidModularSolver {
public:
    using FluidModularSolver::FluidModularSolver;  // 继承构造

    void solve(FluidContext& ctx, Real dt) final {
        solve(static_cast<CPUFluidContext&>(ctx), dt);
    }

    /// CPU Solver 实现这个——直接拿到强类型 Context
    virtual void solve(CPUFluidContext& ctx, Real dt) = 0;
};

} // namespace fluid::cpu
